use std::process::ExitCode;

use strand_decode_kernel::block_walk::gate_proto::machine_stamp;
use strand_decode_kernel::silence::{BlockClass, SilenceMask};
use strand_quant::codebook::codebook_lut;
use strand_quant::decode::decode_tensor_fixed;
use strand_quant::encode::{compute_block_entropy, encode_tensor_with, extract_block_symbols, EncodeOpts};
use strand_quant::TrellisConfig;

fn mixed_weights(n: usize, block_len: usize, seed: u64) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let b = i / block_len;
            if b % 2 == 1 {
                ((i as f32 + seed as f32) * 0.00031).sin() * 2e-4
            } else {
                ((i as f32 + seed as f32) * 0.0137).sin() * 0.5
            }
        })
        .collect()
}

fn run_gate() -> Result<(), String> {
    std::env::set_var("STRAND_NO_GPU", "1");

    println!("gate-entropy-coupling: checking IDENTITY LAW...");

    let configs: Vec<(TrellisConfig, &str)> = vec![
        (TrellisConfig::for_bpw(3.0), "k3 L7 (3-bit deploy)"),
        (TrellisConfig::for_bpw(2.0), "k2 L6"),
        (TrellisConfig::for_bpw(4.0), "k4 L8"),
        (TrellisConfig::for_bpw_l(2.0, 12), "k2 L12 (2-bit reopen)"),
    ];

    let opts_default = EncodeOpts::default();
    let opts_psi_zero = EncodeOpts { entropy_bonus_scale: 0.0, entropy_bonus_two_pass: false, ..Default::default() };

    let mut id_checked = 0usize;
    for (cfg, label) in &configs {
        for &n in &[2048usize, 512, 257] {
            for seed in 0u64..3 {
                let w = mixed_weights(n, cfg.block_len, seed);

                let enc_default = encode_tensor_with(&w, cfg, &opts_default);
                let enc_psi_zero = encode_tensor_with(&w, cfg, &opts_psi_zero);

                if enc_default != enc_psi_zero {
                    return Err(format!("IDENTITY FAIL: psi_scale=0.0 != default for {label} n={n} seed={seed}"));
                }

                let dec_default = decode_tensor_fixed(&enc_default, cfg);
                let dec_psi = decode_tensor_fixed(&enc_psi_zero, cfg);
                if dec_default != dec_psi {
                    return Err(format!("DECODE IDENTITY FAIL: {label} n={n} seed={seed}"));
                }

                id_checked += 1;
            }
        }
    }
    println!("gate-entropy-coupling IDENTITY: PASS ({id_checked} cells, psi_scale=0.0 is byte-identical to default)");

    println!("\ngate-entropy-coupling COMPRESSIBILITY DISTRIBUTION:");
    println!("  (config: k3 L7, n=2048 weights, mixed near-zero/varied tensor, seed=42)");

    let cfg_demo = TrellisConfig::for_bpw(3.0);
    let w_demo = mixed_weights(2048, cfg_demo.block_len, 42);
    let enc_demo = encode_tensor_with(&w_demo, &cfg_demo, &opts_default);

    let n_blocks = enc_demo.blocks.len();
    let mut compressibilities = Vec::with_capacity(n_blocks);
    for b in 0..n_blocks {
        let syms = extract_block_symbols(&enc_demo, b, &cfg_demo);
        let c = compute_block_entropy(&syms, cfg_demo.k_bits as u8);
        compressibilities.push(c);
    }

    println!("  block  block_type  compressibility");
    for (b, &c) in compressibilities.iter().enumerate() {
        let btype = if enc_demo.blocks[b].n < cfg_demo.block_len as u32 {
            "partial"
        } else if b % 2 == 1 {
            "near-zero"
        } else {
            "varied"
        };
        println!("  {:>5}  {:<10}  {:.6}", b, btype, c);
    }

    let sum_c: f64 = compressibilities.iter().sum();
    let mean_c = sum_c / n_blocks as f64;
    let min_c = compressibilities.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_c = compressibilities.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    println!("  summary: n_blocks={} mean={:.4} min={:.4} max={:.4}", n_blocks, mean_c, min_c, max_c);

    println!("\ngate-entropy-coupling SILENCE DELTA  (psi_scale=0.0 vs psi_scale=1e-4 two-pass):");
    println!("{:<26} {:>8} {:>8} {:>8} {:>10}", "config", "n_blocks", "sil_base", "sil_psi", "delta");

    let opts_psi = EncodeOpts { entropy_bonus_scale: 1e-4, entropy_bonus_two_pass: true, ..Default::default() };
    let opts_base = EncodeOpts::default();

    let mut total_blocks = 0usize;
    let mut total_sil_base = 0usize;
    let mut total_sil_psi = 0usize;

    for (cfg, label) in &configs {
        let lut = codebook_lut(cfg.l_bits);
        let n = 4096usize;
        let w = mixed_weights(n, cfg.block_len, 42);

        let enc_base = encode_tensor_with(&w, cfg, &opts_base);
        let enc_psi = encode_tensor_with(&w, cfg, &opts_psi);

        let mask_base = SilenceMask::build(&enc_base, cfg, lut);
        let mask_psi = SilenceMask::build(&enc_psi, cfg, lut);

        let dec_psi = decode_tensor_fixed(&enc_psi, cfg);
        let mut w_off = 0usize;
        for (b, blk) in enc_psi.blocks.iter().enumerate() {
            let blen = blk.n as usize;
            if let Some(consts) = &mask_psi.consts[b] {
                let vals = &dec_psi[w_off..w_off + blen];
                for (sb, &c) in consts.iter().enumerate() {
                    let s = sb * strand_quant::encode::SUB_BLOCK;
                    let e = (s + strand_quant::encode::SUB_BLOCK).min(blen);
                    for &v in &vals[s..e] {
                        if v != c {
                            return Err(format!("PSI DECODE CONSTANT MISMATCH: {label} block {b} sb {sb} c={c} v={v}"));
                        }
                    }
                }
            }
            w_off += blen;
        }

        let sil_base = mask_base.class.iter().filter(|&&c| c == BlockClass::SilentZero).count();
        let sil_psi = mask_psi.class.iter().filter(|&&c| c == BlockClass::SilentZero).count();
        let nb = enc_base.blocks.len();

        println!("{:<26} {:>8} {:>8} {:>8} {:>+10}", label, nb, sil_base, sil_psi, sil_psi as i64 - sil_base as i64);

        total_blocks += nb;
        total_sil_base += sil_base;
        total_sil_psi += sil_psi;
    }

    println!("{:<26} {:>8} {:>8} {:>8} {:>+10}", "TOTAL", total_blocks, total_sil_base, total_sil_psi, total_sil_psi as i64 - total_sil_base as i64,);

    let rate_base = 100.0 * total_sil_base as f64 / total_blocks.max(1) as f64;
    let rate_psi = 100.0 * total_sil_psi as f64 / total_blocks.max(1) as f64;
    println!("silence rate: {:.4}% (base) → {:.4}% (psi_scale=1e-4 two-pass)  delta={:+.4}%", rate_base, rate_psi, rate_psi - rate_base,);

    println!("\ngate-entropy-coupling ONE-PASS ROLLING sanity:");
    let opts_psi_onepass = EncodeOpts { entropy_bonus_scale: 1e-4, entropy_bonus_two_pass: false, ..Default::default() };

    let cfg_op = TrellisConfig::for_bpw(3.0);
    let lut_op = codebook_lut(cfg_op.l_bits);
    let w_op = mixed_weights(2048, cfg_op.block_len, 7);
    let enc_op = encode_tensor_with(&w_op, &cfg_op, &opts_psi_onepass);
    let dec_op = decode_tensor_fixed(&enc_op, &cfg_op);
    if dec_op.len() != w_op.len() {
        return Err(format!("ONE-PASS DECODE LEN MISMATCH: {} vs {}", dec_op.len(), w_op.len()));
    }

    let mask_op = SilenceMask::build(&enc_op, &cfg_op, lut_op);
    let mut w_off2 = 0usize;
    for (b, blk) in enc_op.blocks.iter().enumerate() {
        let blen = blk.n as usize;
        if let Some(consts) = &mask_op.consts[b] {
            let vals = &dec_op[w_off2..w_off2 + blen];
            for (sb, &c) in consts.iter().enumerate() {
                let s = sb * strand_quant::encode::SUB_BLOCK;
                let e = (s + strand_quant::encode::SUB_BLOCK).min(blen);
                for &v in &vals[s..e] {
                    if v != c {
                        return Err(format!("ONE-PASS SILENCE CONSTANT MISMATCH: block {b} sb {sb} c={c} v={v}"));
                    }
                }
            }
        }
        w_off2 += blen;
    }
    println!("  one-pass rolling: PASS (decode len={}, silence mask self-consistent)", dec_op.len());

    println!();
    println!("NOTE: entropy_bonus_scale > 0.0 CHANGES THE EMITTED BITS (expected, by design).");
    println!("      Same licence as SCHISM Δ (silence_bonus) and STRAND_F32_METRIC: bit-changing,");
    println!("      off by default. All KATs and model roots under default opts remain valid.");
    println!("      Ψ requires its own A/B (PPL measurement) before any production adoption.");

    println!("\ngate-entropy-coupling: PASS");
    println!("{}", machine_stamp());
    Ok(())
}

fn main() -> ExitCode {
    match run_gate() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("gate-entropy-coupling FAIL: {e}");
            ExitCode::from(101)
        }
    }
}
