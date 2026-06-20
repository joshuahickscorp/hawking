
use std::process::ExitCode;

use strand_decode_kernel::block_walk::gate_proto::machine_stamp;
use strand_decode_kernel::silence::{BlockClass, SilenceMask};
use strand_quant::codebook::codebook_lut;
use strand_quant::decode::decode_tensor_fixed;
use strand_quant::encode::{encode_tensor_with, EncodeOpts};
use strand_quant::TrellisConfig;

fn test_weights(n: usize, block_len: usize, b_near_zero: usize, seed: u64) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let b = i / block_len;
            if b == b_near_zero {
                
                ((i as f32 + seed as f32) * 0.00031).sin() * 2e-4
            } else {
                ((i as f32 + seed as f32) * 0.0137).sin() * 0.5
            }
        })
        .collect()
}

fn run_gate() -> Result<(), String> {
    
    std::env::set_var("STRAND_NO_GPU", "1");

    let configs: Vec<(TrellisConfig, &str)> = vec![
        (TrellisConfig::for_bpw(3.0), "k3 L7 (3-bit deploy)"),
        (TrellisConfig::for_bpw(2.0), "k2 L6"),
        (TrellisConfig::for_bpw(4.0), "k4 L8"),
        (TrellisConfig::for_bpw_l(2.0, 12), "k2 L12 (2-bit reopen)"),
    ];

    let opts_bonus0 = EncodeOpts { silence_bonus: 0.0, ..Default::default() };
    let opts_no_bonus = EncodeOpts::default();

    let mut id_checked = 0usize;
    for (cfg, label) in &configs {
        let lut = codebook_lut(cfg.l_bits);
        for &n in &[2048usize, 512, 257] {
            for seed in 0u64..3 {
                let w = test_weights(n, cfg.block_len, 1, seed);

                let enc_default = encode_tensor_with(&w, cfg, &opts_no_bonus);
                let enc_bonus0 = encode_tensor_with(
                    &w,
                    cfg,
                    &EncodeOpts { silence_bonus: 0.0, ..Default::default() },
                );

                if enc_default != enc_bonus0 {
                    return Err(format!(
                        "IDENTITY FAIL: bonus=0.0 != default for {label} n={n} seed={seed}"
                    ));
                }

                let decoded_d = decode_tensor_fixed(&enc_default, cfg);
                let decoded_b = decode_tensor_fixed(&enc_bonus0, cfg);
                if decoded_d != decoded_b {
                    return Err(format!(
                        "DECODE IDENTITY FAIL: {label} n={n} seed={seed}"
                    ));
                }

                let mask_d = SilenceMask::build(&enc_default, cfg, lut);
                let mask_b = SilenceMask::build(&enc_bonus0, cfg, lut);
                if mask_d.class != mask_b.class {
                    return Err(format!(
                        "SILENCE MASK MISMATCH: bonus=0.0 vs default {label} n={n} seed={seed}"
                    ));
                }

                id_checked += 1;
            }
        }
    }

    println!(
        "gate-silence-bonus IDENTITY: PASS ({id_checked} cells, bonus=0.0 is byte-identical to default)"
    );

    let bonus_val = 1e-5f64;
    let opts_bonus = EncodeOpts { silence_bonus: bonus_val, ..Default::default() };

    println!("\ngate-silence-bonus CENSUS  (bonus=0.0 vs bonus=1e-5):");
    println!("{:<26} {:>8} {:>8} {:>10}", "config", "sil0_base", "sil0_bonus", "delta");

    let mut total_base = 0usize;
    let mut total_bonus = 0usize;
    let mut total_blocks = 0usize;

    for (cfg, label) in &configs {
        let lut = codebook_lut(cfg.l_bits);
        
        let n = 4096usize;
        let w = test_weights(n, cfg.block_len, 1, 42);

        let enc_base = encode_tensor_with(&w, cfg, &opts_bonus0);
        let enc_bonus = encode_tensor_with(&w, cfg, &opts_bonus);

        let decoded_base = decode_tensor_fixed(&enc_base, cfg);
        let decoded_bonus_raw = decode_tensor_fixed(&enc_bonus, cfg);
        
        let mask_base = SilenceMask::build(&enc_base, cfg, lut);
        let mask_bonus = SilenceMask::build(&enc_bonus, cfg, lut);

        let mut w_off = 0usize;
        for (b, blk) in enc_bonus.blocks.iter().enumerate() {
            let blen = blk.n as usize;
            if let Some(consts) = &mask_bonus.consts[b] {
                
                let vals = &decoded_bonus_raw[w_off..w_off + blen];
                for (sb, &c) in consts.iter().enumerate() {
                    let s = sb * strand_quant::encode::SUB_BLOCK;
                    let e = (s + strand_quant::encode::SUB_BLOCK).min(blen);
                    for &v in &vals[s..e] {
                        if v != c {
                            return Err(format!(
                                "BONUS SILENCE CONSTANT MISMATCH: {label} block {b} sb {sb} c={c} v={v}"
                            ));
                        }
                    }
                }
            }
            w_off += blen;
        }

        let sil0_base = mask_base.class.iter().filter(|&&c| c == BlockClass::SilentZero).count();
        let sil0_bonus = mask_bonus.class.iter().filter(|&&c| c == BlockClass::SilentZero).count();
        let n_blocks = enc_base.blocks.len();

        let _ = decoded_base; 

        println!(
            "{:<26} {:>8} {:>8} {:>+10}",
            label,
            sil0_base,
            sil0_bonus,
            sil0_bonus as i64 - sil0_base as i64
        );

        total_base += sil0_base;
        total_bonus += sil0_bonus;
        total_blocks += n_blocks;
    }

    println!(
        "{:<26} {:>8} {:>8} {:>+10}",
        "TOTAL",
        total_base,
        total_bonus,
        total_bonus as i64 - total_base as i64,
    );

    let rate_base = 100.0 * total_base as f64 / total_blocks.max(1) as f64;
    let rate_bonus = 100.0 * total_bonus as f64 / total_blocks.max(1) as f64;
    println!(
        "silence rate: {:.4}% (base) → {:.4}% (bonus=1e-5)  delta={:+.4}%",
        rate_base,
        rate_bonus,
        rate_bonus - rate_base,
    );

    if total_bonus < total_base {
        return Err(format!(
            "REGRESSION: bonus=1e-5 gave FEWER silent blocks ({total_bonus}) than base ({total_base})"
        ));
    }

    println!("\ngate-silence-bonus: PASS");
    println!("{}", machine_stamp());
    Ok(())
}

fn main() -> ExitCode {
    match run_gate() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("gate-silence-bonus FAIL: {e}");
            ExitCode::from(101)
        }
    }
}
