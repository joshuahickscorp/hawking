#[cfg(not(target_os = "macos"))]
fn main() {
    println!("gate-bitslice: Metal is macOS-only; nothing to gate on this target.");
}

#[cfg(target_os = "macos")]
fn main() {
    macos::run();
}

#[cfg(target_os = "macos")]
mod macos {
    use std::time::Instant;

    use strand_decode_kernel::block_walk::gate_proto::{canonical_configs, machine_stamp, synth_encoded};
    use strand_decode_kernel::gemv_par::decode_q12_par;
    use strand_decode_kernel::metal::{bake_bitslice_entries, bake_bitslice_entries_vec, bitslice_decode_q12, bitslice_decode_q12_computed, BitsliceGpu, StrandGpu};
    use strand_decode_kernel::neon_lut::decode_q12_neonlut;
    use strand_quant::codebook::codebook_lut;
    use strand_quant::decode::{decode_tensor_fixed, decode_tensor_fixed_with_lut_vec};
    use strand_quant::encode::{encode_tensor, encode_tensor_with, vector_lut_from_scalar, EncodeOpts};
    use strand_quant::TrellisConfig;

    fn qat_running() -> bool {
        std::process::Command::new("pgrep").args(["-f", "strand-qat"]).output().map(|o| o.status.success()).unwrap_or(false)
    }

    fn identity_matrix(gpu: &BitsliceGpu) -> usize {
        let mut checked = 0usize;
        for (cfg, label) in canonical_configs() {
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
                    let got = bitslice_decode_q12(gpu, enc, &cfg);
                    assert_eq!(
                        got, reference,
                        "IDENTITY VIOLATION: bitslice GPU diverged from decode_tensor_fixed \
                         at {label}, n={n}, seed={seed}, tail={}, affine={} — release blocker",
                        enc.tail_biting, enc.has_affine_min
                    );
                    // T3: computed-codebook GPU kernel (inline integer Acklam, no LUT
                    // gather / staging) must be byte-identical to BOTH the CPU
                    // decode_tensor_fixed AND the NEON twin — every cell, every weight.
                    let neon = decode_q12_neonlut(enc, &cfg);
                    assert_eq!(
                        neon, reference,
                        "IDENTITY VIOLATION: NEON decode_q12_neonlut diverged from \
                         decode_tensor_fixed at {label}, n={n}, seed={seed}, tail={}, affine={}",
                        enc.tail_biting, enc.has_affine_min
                    );
                    let computed = bitslice_decode_q12_computed(gpu, enc, &cfg);
                    assert_eq!(
                        computed, reference,
                        "IDENTITY VIOLATION: computed-codebook GPU kernel diverged from \
                         decode_tensor_fixed (CPU) at {label}, n={n}, seed={seed}, tail={}, \
                         affine={} — T3 release blocker",
                        enc.tail_biting, enc.has_affine_min
                    );
                    assert_eq!(
                        computed, neon,
                        "IDENTITY VIOLATION: computed-codebook GPU kernel diverged from NEON \
                         at {label}, n={n}, seed={seed}, tail={}, affine={} — T3 release blocker",
                        enc.tail_biting, enc.has_affine_min
                    );
                    checked += 1;
                }
            }
        }

        {
            use strand_decode_kernel::metal::bitslice_decode_q12_with_lut;
            use strand_quant::decode::decode_tensor_fixed_with_lut;
            let cfg = TrellisConfig::for_bpw(3.0).with_vec_dim(2);
            let (ns, d) = (cfg.num_states(), cfg.vec_dim());
            let lut: Vec<i32> = (0..ns * d).map(|i| ((i as u32).wrapping_mul(2654435761) >> 20) as i32 - 2048).collect();
            let w: Vec<f32> = (0..700).map(|i| ((i as f32) * 0.011).cos() * 0.3).collect();
            let enc = encode_tensor(&w, &cfg);
            let want = decode_tensor_fixed_with_lut(&enc, &cfg, &lut);
            assert_eq!(bitslice_decode_q12_with_lut(gpu, &enc, &cfg, &lut), want, "bitslice vec fallback");
            checked += 1;
        }
        checked
    }

    fn bench(gpu: &BitsliceGpu, strand_gpu: Option<&StrandGpu>) {
        let (out_f, in_f) = (18944usize, 3584usize);
        let total = out_f * in_f;
        println!("\n== bench (best-of-10 GPU dispatch, ffn_down {out_f}x{in_f} = {:.1}M weights) ==", total as f64 / 1e6);
        println!("  {}", machine_stamp());

        let peak = strand_gpu.map(|g| g.bench_peak_bw(64 << 20, 5)).unwrap_or(f64::NAN);
        println!("  measured streaming peak: {:.1} GB/s (grid-stride f32 sum, 256 MB)", peak / 1e9);

        for (cfg, label) in [(TrellisConfig::for_bpw(3.0), "3-bit deploy (k3 L7, 512 B TG LUT)"), (TrellisConfig::for_bpw_l(2.0, 12), "2-bit reopen (k2 L12, 16 KB TG LUT)")] {
            println!("  -- {label} --");
            let enc = synth_encoded(total, cfg.k_bits, 256);
            let lut = codebook_lut(cfg.l_bits);
            let tbl = bake_bitslice_entries(&enc, &cfg).expect("256-weight blocks bake");

            let got = gpu.decode_q12(&enc.bits, &tbl, lut, total, cfg.k_bits, cfg.l_bits);
            let want = decode_tensor_fixed(&enc, &cfg);
            assert_eq!(got, want, "bench-shape identity violated at {label}");

            let dt = gpu.bench_decode(&enc.bits, &tbl, lut, total, cfg.k_bits, cfg.l_bits, 10);
            let gws = total as f64 / dt / 1e9;

            let bytes = (total as f64) * (cfg.k_bits as f64) / 8.0 + (tbl.len() * 80) as f64 + (total * 4) as f64;
            let achieved = bytes / dt;
            let pct = 100.0 * achieved / peak;

            let mut cpu_best = f64::INFINITY;
            for _ in 0..3 {
                let t = Instant::now();
                let out = decode_q12_par(&enc, &cfg);
                cpu_best = cpu_best.min(t.elapsed().as_secs_f64());
                std::hint::black_box(&out);
            }
            let cpu_gws = total as f64 / cpu_best / 1e9;

            println!("  GPU bitslice : {:>7.2} ms   {:>6.2} Gw/s   {:>6.1} GB/s moved = {:>5.1}% of peak", dt * 1e3, gws, achieved / 1e9, pct);
            println!("  CPU rayon par: {:>7.2} ms   {:>6.2} Gw/s   (12-core baseline)   GPU/CPU = {:.2}x", cpu_best * 1e3, cpu_gws, gws / cpu_gws);
            let verdict = if pct >= 50.0 {
                "REVIVAL (>=50% peak) — fused y+=W.x stretch is unlocked"
            } else if pct < 35.0 {
                "below the 35% close line for this variant"
            } else {
                "between 35% and 50% — variant sweeps decide"
            };
            println!("  verdict @ {label}: {verdict}");

            let x: Vec<f32> = (0..in_f).map(|i| (i as f32 * 0.05).sin()).collect();
            let y_gpu = gpu.matvec(&enc.bits, &tbl, lut, out_f as u32, in_f as u32, cfg.k_bits, cfg.l_bits, &x);
            let inv = 1.0f32 / 4096.0;
            for r in (0..out_f).step_by(997) {
                let row = &want[r * in_f..(r + 1) * in_f];
                let mut acc = 0.0f32;
                for i in 0..in_f {
                    acc += (row[i] as f32) * inv * x[i];
                }
                let denom = acc.abs().max(1e-3);
                assert!((y_gpu[r] - acc).abs() / denom < 1e-3, "fused y diverged beyond float tolerance at {label} row {r}: GPU {} vs CPU {acc}", y_gpu[r]);
            }
            let dt_f = gpu.bench_matvec(&enc.bits, &tbl, lut, out_f as u32, in_f as u32, cfg.k_bits, cfg.l_bits, &x, 10);
            let f_gws = total as f64 / dt_f / 1e9;

            let f_bytes = (total as f64) * (cfg.k_bits as f64) / 8.0 + (tbl.len() * 80) as f64 + (tbl.len() * 8) as f64 + ((in_f + out_f) * 4) as f64;
            println!(
                "  GPU fused y=Wx (B=1): {:>7.2} ms   {:>6.2} Gw/s effective   {:>5.1} GB/s moved \
                 ({:.1}% of peak; table is {:.0}% of traffic)",
                dt_f * 1e3,
                f_gws,
                f_bytes / dt_f / 1e9,
                100.0 * (f_bytes / dt_f) / peak,
                100.0 * (tbl.len() * 80) as f64 / f_bytes,
            );
        }
        println!(
            "\n  gate thresholds (roadmap §6, committed before running): >=50% peak = revival; \
             <35% after a fair effort = GPU decode closed for good."
        );
    }

    fn bench_gemm_prompt(gpu: &BitsliceGpu) {
        use strand_decode_kernel::fused::fused_gemm;

        let (out_f, in_f) = (18944usize, 3584usize);
        let total = out_f * in_f;
        println!("\n== fused GEMM (prompt phase): B in {{4,16,64}}, ffn_down {out_f}x{in_f} = {:.1}M weights ==", total as f64 / 1e6);
        println!("  {}", machine_stamp());

        for (cfg, label) in [(TrellisConfig::for_bpw(3.0), "3-bit deploy (k3 L7)"), (TrellisConfig::for_bpw_l(2.0, 12), "2-bit reopen (k2 L12)")] {
            println!("  -- {label} --");
            let enc = synth_encoded(total, cfg.k_bits, 256);
            let lut = codebook_lut(cfg.l_bits);
            let tbl = bake_bitslice_entries(&enc, &cfg).expect("256-weight blocks bake");
            let want = decode_tensor_fixed(&enc, &cfg);

            for &batch in &[4usize, 16, 64] {
                let mut xs = vec![0.0f32; batch * in_f];
                let probe: Vec<usize> = (0..batch).map(|b| (911 * b + 17) % in_f).collect();
                for (b, &c) in probe.iter().enumerate() {
                    xs[b * in_f + c] = 1.0;
                }
                let y = gpu.gemm(&enc.bits, &tbl, lut, out_f as u32, in_f as u32, cfg.k_bits, cfg.l_bits, &xs, batch);
                for r in 0..out_f {
                    for (b, &c) in probe.iter().enumerate() {
                        let recovered = (y[r * batch + b] * 4096.0).round() as i32;
                        let expected = want[r * in_f + c];
                        assert_eq!(recovered, expected, "GEMM one-hot Q12 diverged at {label} B={batch} r={r} c={c} — release blocker");
                    }
                }

                let xs: Vec<f32> = (0..batch * in_f).map(|i| ((i as f32) * 0.031).cos()).collect();
                let dt = gpu.bench_gemm(&enc.bits, &tbl, lut, out_f as u32, in_f as u32, cfg.k_bits, cfg.l_bits, &xs, batch, 5);
                let eff_gws = total as f64 / dt / 1e9;
                let gmacs = total as f64 * batch as f64 / dt / 1e9;
                let per_col_ms = dt / batch as f64 * 1e3;

                let mut cpu_best = f64::INFINITY;
                for _ in 0..3 {
                    let t = Instant::now();
                    let yc = fused_gemm(&enc, &cfg, None, out_f, in_f, &xs, batch);
                    cpu_best = cpu_best.min(t.elapsed().as_secs_f64());
                    std::hint::black_box(&yc);
                }
                let cpu_gmacs = total as f64 * batch as f64 / cpu_best / 1e9;

                println!(
                    "  B={batch:>2}: GPU {:>7.2} ms  {:>6.2} Gw/s eff  {:>6.1} GMAC/s  {:>6.3} ms/col   \
                     CPU fused-NEON {:>7.2} ms ({:>5.1} GMAC/s)   GPU/CPU = {:.2}x",
                    dt * 1e3,
                    eff_gws,
                    gmacs,
                    per_col_ms,
                    cpu_best * 1e3,
                    cpu_gmacs,
                    cpu_best / dt
                );
            }
        }
        println!(
            "  (identity protocol: one-hot batch lanes pin GPU Q12 == decode_tensor_fixed across \
             every row x lane BEFORE perf; float y order documented per kernel, not cross-device.)"
        );
    }

    fn bench_prepared(gpu: &BitsliceGpu) {
        println!("\n== prepared-GPU integration (load-time bake; the token-decode loop) ==");
        println!("  {}", machine_stamp());

        let (out_f, in_f) = (18944usize, 3584usize);
        let total = out_f * in_f;
        for (cfg, label) in [(TrellisConfig::for_bpw(3.0), "3-bit deploy (k3 L7)"), (TrellisConfig::for_bpw_l(2.0, 12), "2-bit reopen (k2 L12)")] {
            let enc = synth_encoded(total, cfg.k_bits, 256);
            let want = decode_tensor_fixed(&enc, &cfg);

            let mut cold_best = f64::INFINITY;
            for _ in 0..3 {
                let t = Instant::now();
                let got = bitslice_decode_q12(gpu, &enc, &cfg);
                cold_best = cold_best.min(t.elapsed().as_secs_f64());
                std::hint::black_box(&got);
            }

            let t_prep = Instant::now();
            let p = gpu.prepare(&enc, &cfg).expect("prepare");
            let prep_s = t_prep.elapsed().as_secs_f64();
            assert_eq!(gpu.decode_q12_prepared(&p), want, "prepared GPU decode diverged at {label} — release blocker");
            let mut disp_best = f64::INFINITY;
            for _ in 0..10 {
                let t = Instant::now();
                gpu.dispatch_prepared(&p);
                let dt = t.elapsed().as_secs_f64();
                if dt > 0.0 && dt < disp_best {
                    disp_best = dt;
                }
            }
            println!(
                "  -- {label} (ffn_down, {:.1}M w) --\n\
                 \x20 cold (bake+upload+dispatch/call): {:>7.2} ms   prepared dispatch: {:>7.2} ms \
                 ({:>5.2} Gw/s)   rebuild tax eliminated: {:.2}x   one-time prepare: {:.0} ms   \
                 resident: {:.1} MB ({:.4} B/w)",
                total as f64 / 1e6,
                cold_best * 1e3,
                disp_best * 1e3,
                total as f64 / disp_best / 1e9,
                cold_best / disp_best,
                prep_s * 1e3,
                p.gpu_bytes() as f64 / 1e6,
                p.gpu_bytes() as f64 / total as f64,
            );
        }

        let shapes: Vec<(usize, usize)> = (0..4).flat_map(|_| [(2048usize, 2048usize), (512, 2048), (512, 2048), (2048, 2048), (5632, 2048), (2048, 5632)]).collect();
        let model_w: usize = shapes.iter().map(|(r, c)| r * c).sum();
        for (cfg, label) in [(TrellisConfig::for_bpw(3.0), "3-bit deploy (k3 L7)"), (TrellisConfig::for_bpw_l(2.0, 12), "2-bit reopen (k2 L12)")] {
            let mut encs = Vec::with_capacity(shapes.len());
            let mut prepared = Vec::with_capacity(shapes.len());
            for &(r, c) in &shapes {
                let enc = synth_encoded(r * c, cfg.k_bits, 256);
                let p = gpu.prepare(&enc, &cfg).expect("prepare model tensor");
                encs.push(enc);
                prepared.push(p);
            }

            gpu.dispatch_prepared_all(&prepared);
            for (p, enc) in prepared.iter().zip(encs.iter()) {
                assert_eq!(p.read_out(), decode_tensor_fixed(enc, &cfg), "batched prepared dispatch diverged at {label} — release blocker");
            }

            let mut one_best = f64::INFINITY;
            for _ in 0..10 {
                let t = Instant::now();
                gpu.dispatch_prepared_all(&prepared);
                let dt = t.elapsed().as_secs_f64();
                if dt > 0.0 && dt < one_best {
                    one_best = dt;
                }
            }
            let mut many_best = f64::INFINITY;
            for _ in 0..10 {
                let t = Instant::now();
                for p in &prepared {
                    gpu.dispatch_prepared(p);
                }
                let dt = t.elapsed().as_secs_f64();
                if dt > 0.0 && dt < many_best {
                    many_best = dt;
                }
            }

            let mut cpu_best = f64::INFINITY;
            for _ in 0..3 {
                let t = Instant::now();
                for enc in &encs {
                    let out = decode_q12_par(enc, &cfg);
                    std::hint::black_box(&out);
                }
                cpu_best = cpu_best.min(t.elapsed().as_secs_f64());
            }

            let resident: usize = prepared.iter().map(|p| p.gpu_bytes()).sum();
            println!(
                "  -- {label} (24 tensors, {:.1}M w/token) --\n\
                 \x20 GPU 1 commit/token : {:>7.2} ms/token  ({:>5.2} Gw/s)\n\
                 \x20 GPU 24 commits/token: {:>7.2} ms/token  (commit overhead ~{:.3} ms/tensor)\n\
                 \x20 CPU rayon           : {:>7.2} ms/token  ({:>5.2} Gw/s)   GPU(1-commit)/CPU = {:.2}x\n\
                 \x20 GPU-resident total: {:.0} MB ({:.4} B/w incl. 4 B/w out buffers)",
                model_w as f64 / 1e6,
                one_best * 1e3,
                model_w as f64 / one_best / 1e9,
                many_best * 1e3,
                (many_best - one_best).max(0.0) / shapes.len() as f64 * 1e3,
                cpu_best * 1e3,
                model_w as f64 / cpu_best / 1e9,
                cpu_best / one_best,
                resident as f64 / 1e6,
                resident as f64 / model_w as f64,
            );
        }
        println!(
            "  (honest framing: decode-primitive numbers — Q12 streams per token, no MAC/attention/\
             sampling. The fused/GEMM kernels carry the MAC story; end-to-end inference is dismantle's.)"
        );
    }

    /// B.7 vector-d=2 gate: (1) byte-identity vs the CPU twin across an edge-length
    /// sweep, then (2) fused-B=1 throughput at (L7,d2)/(L10,d2) vs the scalar L12,d1
    /// incumbent. Returns true iff identity held on every cell (timing is inadmissible
    /// otherwise — caller reports KILL).
    fn vec_d2_gate(gpu: &BitsliceGpu) -> bool {
        const D: usize = 2;
        println!("\n========================================================================");
        println!("== B.7 VECTOR d={D} GATE (decode + fused-B=1) ==");
        println!("  {}", machine_stamp());
        println!("  host sizeof(BitsliceEntry) = {} B; GPU sizeof = {} B (probe)", std::mem::size_of::<strand_decode_kernel::metal::BitsliceEntry>(), gpu.gpu_entry_sizeof(),);

        // ---- (1) IDENTITY GATE FIRST -------------------------------------------------
        // Encode with a vec cfg (implicit codebook = broadcast LUT), decode-reference
        // with decode_tensor_fixed_with_lut_vec using that SAME broadcast LUT, compare
        // to the GPU _vec kernel. Edge-length sweep includes odd n (non-multiples of d),
        // n < d, exact multiples, sub-block boundaries (32/256), and multi-block sizes.
        println!("\n  -- identity: GPU _vec decode == decode_tensor_fixed_with_lut_vec --");
        let id_configs = [
            (TrellisConfig::for_bpw(3.0).with_vec_dim(D as u32), "k3 L7 d2"),
            (TrellisConfig::for_bpw_l(3.0, 10).with_vec_dim(D as u32), "k3 L10 d2"),
            (TrellisConfig::for_bpw_l(2.0, 12).with_vec_dim(D as u32), "k2 L12 d2"),
            (TrellisConfig::for_bpw(2.0).with_vec_dim(D as u32), "k2 L6 d2"),
        ];
        // edge lengths: 1 (n<d), 2,3 (around d), 31..33 (sub-block 32), 255..257 (block 256),
        // 511..513 (multi-block), plus a few odd primes and a big multi-block size.
        let edge_ns: [usize; 18] = [1, 2, 3, 4, 5, 31, 32, 33, 63, 127, 255, 256, 257, 511, 512, 513, 999, 4097];
        let mut checked = 0usize;
        let mut first_fail: Option<String> = None;
        'outer: for (cfg, label) in &id_configs {
            let scalar_lut = codebook_lut(cfg.l_bits);
            let vec_lut = vector_lut_from_scalar(scalar_lut, D); // 2^L * d, broadcast
            for &n in &edge_ns {
                // a couple of phase seeds per (cfg, n) to vary the bit pattern
                for seed in 0..3u64 {
                    let w: Vec<f32> = (0..n).map(|i| ((i as f32 + seed as f32 * 7.0) * 0.0137).sin() * 0.5).collect();
                    // plain + tail-biting + affine-min variants (affine exercises off[])
                    let variants = [
                        encode_tensor(&w, cfg),
                        encode_tensor_with(&w, cfg, &EncodeOpts { tail_biting: true, ..Default::default() }),
                        encode_tensor_with(&w, cfg, &EncodeOpts { affine_min: true, ..Default::default() }),
                        encode_tensor_with(&w, cfg, &EncodeOpts { tail_biting: true, affine_min: true, ..Default::default() }),
                    ];
                    for enc in &variants {
                        let reference = decode_tensor_fixed_with_lut_vec(enc, cfg, &vec_lut);
                        let Some(tbl) = bake_bitslice_entries_vec(enc, cfg) else {
                            first_fail = Some(format!("{label} n={n} seed={seed}: bake_bitslice_entries_vec returned None"));
                            break 'outer;
                        };
                        let got = gpu.decode_q12_vec(&enc.bits, &tbl, &vec_lut, enc.total, cfg.k_bits, cfg.l_bits, D);
                        if got != reference {
                            // locate first divergent index for evidence
                            let idx = got.iter().zip(reference.iter()).position(|(a, b)| a != b).unwrap_or(usize::MAX);
                            let (g, r) = if idx < got.len() { (got[idx], reference[idx]) } else { (0, 0) };
                            first_fail = Some(format!(
                                "IDENTITY VIOLATION at {label}, n={n}, seed={seed}, tail={}, affine={}: \
                                 first diff @ i={idx} GPU={g} vs CPU={r} (len got={} ref={})",
                                enc.tail_biting,
                                enc.has_affine_min,
                                got.len(),
                                reference.len(),
                            ));
                            break 'outer;
                        }
                        checked += 1;
                    }
                }
            }
            println!("    {label}: OK across {} edge×variant cells", edge_ns.len() * 3 * 4);
        }

        if let Some(msg) = first_fail {
            println!("\n  !! {msg}");
            println!("  IDENTITY GATE FAILED — timing is inadmissible. VERDICT: KILL");
            return false;
        }
        println!("  identity: {checked} GPU _vec cells byte-identical to decode_tensor_fixed_with_lut_vec ✓");

        // ---- (2) FUSED B=1 BENCH -----------------------------------------------------
        let (out_f, in_f) = (18944usize, 3584usize);
        let total = out_f * in_f;
        // measured streaming peak (same probe the scalar bench uses)
        let peak = StrandGpu::new().map(|g| g.bench_peak_bw(64 << 20, 5)).unwrap_or(f64::NAN);
        println!("\n  -- fused y=Wx (B=1), ffn_down {out_f}x{in_f} = {:.1}M weights --", total as f64 / 1e6);
        println!("  measured streaming peak: {:.1} GB/s", peak / 1e9);

        // scalar L12,d1 incumbent (apples-to-apples, same run)
        let scalar_gws = {
            let cfg = TrellisConfig::for_bpw_l(2.0, 12); // k2 L12 d1
            let enc = synth_encoded(total, cfg.k_bits, 256);
            let lut = codebook_lut(cfg.l_bits);
            let tbl = bake_bitslice_entries(&enc, &cfg).expect("scalar bake");
            let x: Vec<f32> = (0..in_f).map(|i| (i as f32 * 0.05).sin()).collect();
            let dt = gpu.bench_matvec(&enc.bits, &tbl, lut, out_f as u32, in_f as u32, cfg.k_bits, cfg.l_bits, &x, 10);
            let gws = total as f64 / dt / 1e9;
            let bytes = (total as f64) * (cfg.k_bits as f64) / 8.0 + (tbl.len() * std::mem::size_of::<strand_decode_kernel::metal::BitsliceEntry>()) as f64 + ((in_f + out_f) * 4) as f64;
            let pct = 100.0 * (bytes / dt) / peak;
            println!("  SCALAR L12,d1 (incumbent): {:>7.3} ms   {:>6.2} Gw/s   {:>5.1}% of peak", dt * 1e3, gws, pct);
            gws
        };

        // vector cells: (L7,d2) and (L10,d2). k=3 (3-bit deploy lineage).
        let mut vec_results: Vec<(String, f64, f64)> = Vec::new();
        for (cfg, label) in [(TrellisConfig::for_bpw(3.0).with_vec_dim(D as u32), "L7,d2"), (TrellisConfig::for_bpw_l(3.0, 10).with_vec_dim(D as u32), "L10,d2")] {
            let enc = synth_encoded(total, cfg.k_bits, 256);
            let scalar_lut = codebook_lut(cfg.l_bits);
            let vec_lut = vector_lut_from_scalar(scalar_lut, D);
            let tbl = bake_bitslice_entries_vec(&enc, &cfg).expect("vec bake");
            let x: Vec<f32> = (0..in_f).map(|i| (i as f32 * 0.05).sin()).collect();
            let dt = gpu.bench_matvec_vec(&enc.bits, &tbl, &vec_lut, out_f as u32, in_f as u32, cfg.k_bits, cfg.l_bits, D, &x, 10);
            // effective weights = true outputs decoded = total (d outputs per symbol)
            let gws = total as f64 / dt / 1e9;
            // bytes moved: payload uses ceil(n/d)*k bits (half the symbols), + table + x/y
            let bytes = (total as f64 / D as f64) * (cfg.k_bits as f64) / 8.0 + (tbl.len() * std::mem::size_of::<strand_decode_kernel::metal::BitsliceEntry>()) as f64 + ((in_f + out_f) * 4) as f64;
            let pct = 100.0 * (bytes / dt) / peak;
            let tg_lut_kb = (1usize << cfg.l_bits) * D * 4;
            println!("  VECTOR {label}: {:>7.3} ms   {:>6.2} Gw/s   {:>5.1}% of peak   (TG LUT {} B)", dt * 1e3, gws, pct, tg_lut_kb,);
            vec_results.push((label.to_string(), gws, pct));
        }

        // ---- VERDICT -----------------------------------------------------------------
        println!(
            "\n  -- VERDICT (committed before run: PASS = identity exact AND fused-B=1 >=1.3x \
                 scalar incumbent; MARGINAL 1.1-1.3x; KILL <1.1x or identity break) --"
        );
        let best = vec_results.iter().cloned().fold(("none".to_string(), 0.0f64, 0.0f64), |acc, x| if x.1 > acc.1 { x } else { acc });
        for (label, gws, _pct) in &vec_results {
            let ratio = gws / scalar_gws;
            let v = if *gws / scalar_gws >= 1.3 {
                "PASS"
            } else if ratio >= 1.1 {
                "MARGINAL"
            } else {
                "KILL"
            };
            println!("  {label}: {:.2} Gw/s  /  scalar {:.2} Gw/s = {:.2}x  => {v}", gws, scalar_gws, ratio);
        }
        let best_ratio = best.1 / scalar_gws;
        let overall = if best_ratio >= 1.3 {
            "PASS"
        } else if best_ratio >= 1.1 {
            "MARGINAL"
        } else {
            "KILL"
        };
        println!("  BEST vector cell: {} @ {:.2} Gw/s = {:.2}x scalar  =>  OVERALL VERDICT: {overall}", best.0, best.1, best_ratio);
        true
    }

    pub fn run() {
        let no_wait = std::env::args().any(|a| a == "--no-wait");
        if !no_wait {
            while qat_running() {
                println!(
                    "strand-qat MPS job is running — waiting 60 s before any GPU dispatch \
                     (pass --no-wait to override)…"
                );
                std::thread::sleep(std::time::Duration::from_secs(60));
            }
        }

        let Some(gpu) = BitsliceGpu::new() else {
            println!("gate-bitslice: no Metal device / shader compile failed — nothing to gate.");
            return;
        };
        println!("gate-bitslice — G4 bitslice decode (probe: GPU sizeof(BitsliceEntry) == host {} ✓)", std::mem::size_of::<strand_decode_kernel::metal::BitsliceEntry>());

        let vec_only = std::env::args().any(|a| a == "--vec-only");

        if vec_only {
            // B.7 vector-d=2 gate only (clean output for the integration decision).
            let _ = vec_d2_gate(&gpu);
            return;
        }

        let t = Instant::now();
        let checked = identity_matrix(&gpu);
        println!("identity: {} GPU config×variant cells byte-identical to decode_tensor_fixed in {:.1}s ✓", checked, t.elapsed().as_secs_f64());

        let strand_gpu = StrandGpu::new();
        bench(&gpu, strand_gpu.as_ref());

        bench_gemm_prompt(&gpu);
        bench_prepared(&gpu);

        // B.7 vector-d=2 gate (decode identity + fused-B=1 throughput).
        let _ = vec_d2_gate(&gpu);
    }
}
