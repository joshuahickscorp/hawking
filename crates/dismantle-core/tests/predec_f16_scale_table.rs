//! 1.2 CPU-half scaffold test: the f16 predec scale table widens back to within
//! the f16 precision budget of the f32 table, validating f16 is adequate for
//! Q4_K sub-block scale magnitudes (the bandwidth-cut premise of lever 1.2).
//! Pure CPU (no Metal). The kernel that consumes the f16 table is GPU-lane.

#![cfg(target_os = "macos")]

use dismantle_core::kernels::{predecode_q4_k_scale_table, predecode_q4_k_scale_table_f16};
use half::f16;

/// One row of 64 Q4_K blocks with realistic header scales (d ~0.01, dmin small)
/// and deterministic packed sub-block bytes.
fn make_q4k_bytes(n_blocks: usize) -> Vec<u8> {
    let mut bytes = vec![0u8; n_blocks * 144];
    for b in 0..n_blocks {
        let off = b * 144;
        let d = 0.012_f32 + (b % 7) as f32 * 0.001;
        let dmin = ((b % 5) as f32 - 2.0) * 0.002;
        bytes[off..off + 2].copy_from_slice(&f16::from_f32(d).to_bits().to_le_bytes());
        bytes[off + 2..off + 4].copy_from_slice(&f16::from_f32(dmin).to_bits().to_le_bytes());
        for i in 4..144 {
            bytes[off + i] = ((i * 31 + b * 17) & 0xFF) as u8;
        }
    }
    bytes
}

#[test]
fn f16_predec_table_matches_f32_within_budget() {
    let n_blocks = 64;
    let bytes = make_q4k_bytes(n_blocks);
    let f32_tab = predecode_q4_k_scale_table(&bytes);
    let f16_tab = predecode_q4_k_scale_table_f16(&bytes);

    assert_eq!(f32_tab.len(), n_blocks * 16);
    assert_eq!(f16_tab.len(), f32_tab.len());

    let mut max_abs = 0.0_f32;
    let mut max_rel = 0.0_f32;
    for (&a, h) in f32_tab.iter().zip(f16_tab.iter()) {
        let w = h.to_f32();
        let abs = (a - w).abs();
        max_abs = max_abs.max(abs);
        if a.abs() > 1e-4 {
            max_rel = max_rel.max(abs / a.abs());
        }
    }
    // f16 has an ~11-bit mantissa → relative error < ~5e-4 for in-range values.
    assert!(max_abs < 1e-2, "max abs diff {max_abs} too large for f16 scales");
    assert!(max_rel < 1e-2, "max rel diff {max_rel} exceeds the f16 precision budget");
}
