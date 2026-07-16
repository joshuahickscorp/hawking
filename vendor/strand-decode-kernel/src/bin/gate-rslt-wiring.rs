use strand_decode_kernel::block_walk::gate_proto::{canonical_configs, machine_stamp, synth_encoded};
use strand_decode_kernel::gemv_par::{decode_q12_par, decode_q12_par_counted};
use strand_decode_kernel::prepared::{decode_q12_par_prepared, decode_q12_par_prepared_counted, PreparedTensor};
use strand_quant::encode::{encode_tensor, encode_tensor_with, EncodeOpts};
use strand_quant::TrellisConfig;

fn main() {
    println!("=== gate-rslt-wiring ===");
    println!("{}", machine_stamp());

    let mut total_blocks = 0u64;
    let mut total_decode_calls = 0u64;
    let mut checks = 0usize;

    println!("\n-- gemv_par (decode_q12_par_counted) --");
    for (cfg, label) in canonical_configs() {
        for seed in 0..8u64 {
            let n = 256 + (seed as usize * 337) % 3000;
            let w: Vec<f32> = (0..n).map(|i| ((i as f32 + seed as f32) * 0.0137).sin() * 0.5).collect();
            for opts in [EncodeOpts::default(), EncodeOpts { tail_biting: true, ..Default::default() }, EncodeOpts { affine_min: true, ..Default::default() }] {
                let enc = encode_tensor_with(&w, &cfg, &opts);
                let nb = enc.blocks.len();

                let zero_counts = vec![0u32; nb];
                let q_no_count = decode_q12_par(&enc, &cfg);
                total_decode_calls += 1;

                let mut counts = vec![0u32; nb];
                let q_counted = decode_q12_par_counted(&enc, &cfg, Some(&mut counts));
                total_decode_calls += 1;
                total_blocks += nb as u64;

                assert_eq!(
                    q_counted, q_no_count,
                    "IDENTITY VIOLATION: counted decode diverged from uncounted at {label} \
                     n={n} seed={seed} tail={} affine={}",
                    opts.tail_biting, opts.affine_min
                );

                for (b, &c) in counts.iter().enumerate() {
                    assert!(
                        c > 0,
                        "COUNTS VIOLATION: block {b} count == 0 after counted decode at \
                         {label} n={n} seed={seed}"
                    );
                }

                for (b, &c) in zero_counts.iter().enumerate() {
                    assert_eq!(c, 0, "None-path VIOLATION: zero_counts[{b}] was mutated (should be impossible)");
                }

                checks += 1;
            }
        }
    }
    println!("  gemv_par: {} configs checked OK", checks);

    println!("\n-- prepared (decode_q12_par_prepared_counted) --");
    let prepared_start = checks;
    for (cfg, label) in canonical_configs() {
        for seed in 0..4u64 {
            let n = 512 + (seed as usize * 211) % 2048;
            let w: Vec<f32> = (0..n).map(|i| ((i as f32 + seed as f32) * 0.0091).cos() * 0.4).collect();
            let enc = encode_tensor(&w, &cfg);
            let nb = enc.blocks.len();
            let p = PreparedTensor::new(enc.clone(), cfg.clone());

            let q_prep_base = decode_q12_par_prepared(&p);
            total_decode_calls += 1;

            let mut counts = vec![0u32; nb];
            let q_prep_counted = decode_q12_par_prepared_counted(&p, Some(&mut counts));
            total_decode_calls += 1;
            total_blocks += nb as u64;

            assert_eq!(q_prep_counted, q_prep_base, "IDENTITY VIOLATION: prepared counted decode diverged at {label} n={n} seed={seed}");

            for (b, &c) in counts.iter().enumerate() {
                assert!(c > 0, "COUNTS VIOLATION: prepared block {b} count == 0 at {label} n={n} seed={seed}");
            }

            checks += 1;
        }
    }
    println!("  prepared: {} configs checked OK", checks - prepared_start);

    println!("\n-- saturation --");
    {
        let cfg = TrellisConfig::for_bpw(3.0);
        let n = 256usize;
        let w: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.013).sin() * 0.5).collect();
        let enc = encode_tensor(&w, &cfg);
        let nb = enc.blocks.len();
        let mut counts = vec![u32::MAX - 2; nb];

        for _ in 0..5 {
            decode_q12_par_counted(&enc, &cfg, Some(&mut counts));
        }
        for (b, &c) in counts.iter().enumerate() {
            assert_eq!(c, u32::MAX, "saturation VIOLATION: block {b} count {c} != u32::MAX after overflow");
        }
        println!("  saturation: all {} blocks clamped to u32::MAX OK", nb);
    }

    println!("\n-- large synth tensor (256 blocks) --");
    {
        let cfg = TrellisConfig::for_bpw(3.0);
        let block_len = 256usize;
        let n_blocks_want = 256usize;
        let total = block_len * n_blocks_want;
        let enc = synth_encoded(total, cfg.k_bits, block_len);
        let nb = enc.blocks.len();
        assert_eq!(nb, n_blocks_want);

        let q_ref = decode_q12_par(&enc, &cfg);
        let mut counts = vec![0u32; nb];
        let q_counted = decode_q12_par_counted(&enc, &cfg, Some(&mut counts));

        assert_eq!(q_counted, q_ref, "IDENTITY VIOLATION: large synth tensor");
        assert!(counts.iter().all(|&c| c == 1), "all block counts must equal 1");
        total_blocks += nb as u64;
        total_decode_calls += 2;
        println!("  large synth: {} blocks, identity OK, counts all 1 OK", nb);
    }

    println!("\n=== RSLT wiring gate SUMMARY ===");
    println!("  configs checked : {checks}");
    println!("  total blocks    : {total_blocks}");
    println!("  total decode calls: {total_decode_calls}");
    println!("\ngate PASS");
}
