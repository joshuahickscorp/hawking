#[cfg(test)]
mod tests {
    /// Verify TrellisConfig accepts k_bits=1 via for_bpw(1.0).
    /// for_bpw rounds target to k=1, sets l = (k+4).min(MAX_L) = 5.
    #[test]
    fn bits1_trellis_config_l5() {
        use strand_quant::TrellisConfig;
        let cfg = TrellisConfig::for_bpw(1.0);
        assert_eq!(cfg.k_bits, 1, "k_bits should be 1 for bpw=1.0");
        assert!(cfg.l_bits >= 4, "l_bits must be >= MIN_L=4; got {}", cfg.l_bits);
        assert_eq!(cfg.l_bits, 5, "for_bpw(1.0): l = k+4 = 5");
        assert_eq!(cfg.block_len, 256);
    }

    /// Verify TrellisConfig::new accepts k_bits=1 explicitly.
    /// new() clamps k_bits to clamp(1, MAX_K=4) and l_bits to clamp(MIN_L=4, MAX_L=14),
    /// then enforces l_bits >= k_bits.
    #[test]
    fn bits1_trellis_config_l7_explicit() {
        use strand_quant::TrellisConfig;
        let cfg = TrellisConfig::new(7, 1, 256);
        assert_eq!(cfg.l_bits, 7, "l_bits should be 7");
        assert_eq!(cfg.k_bits, 1, "k_bits should be 1");
        assert_eq!(cfg.block_len, 256, "block_len should be 256");
    }

    #[test]
    #[ignore] // needs release binary; cargo test -p strand-quant -- --ignored
    fn bits1_binary_accepts_flag() {
        // Verify quantize-model binary exits without assertion failure when passed --bits 1
        let bin = std::env::var("STRAND_QUANT_BIN")
            .unwrap_or_else(|_| "target/release/quantize-model".to_string());
        if !std::path::Path::new(&bin).exists() {
            eprintln!("Binary not found at '{}', skipping", bin);
            return;
        }
        let out = std::process::Command::new(&bin)
            .arg("--bits").arg("1").arg("--help")
            .output();
        if let Ok(o) = out {
            let stderr = String::from_utf8_lossy(&o.stderr);
            assert!(
                !stderr.contains("assertion"),
                "Got assertion failure with --bits 1: {}",
                stderr
            );
        }
    }
}
