
#![allow(clippy::needless_range_loop)]

use crate::codebook::{codebook_lut, quantile_lut, QUANTILE_SHIFT};
use crate::decode::{eff_min_q, eff_scale_q, reconstruct_q, WordBitReader, SCALE_SHIFT};
use crate::lut_tables::{FROZEN_MAX_L, FROZEN_MIN_L};
use crate::trellis::read_bits;

const Q_CLAMP: i32 = 6 * (1 << QUANTILE_SHIFT); 

const RECON_ABS_MAX: i64 = ((1i64 << 31) * (Q_CLAMP as i64)) >> SCALE_SHIFT;

fn boundary_scales() -> Vec<i32> {
    let mut v: Vec<i32> = Vec::new();
    for w in 0..=256i64 {
        v.push(w as i32); 
        v.push(-(w as i32)); 
        v.push((i32::MAX as i64 - w) as i32); 
        v.push((i32::MIN as i64 + w) as i32); 
    }
    for p in 0..31u32 {
        let b = 1i64 << p;
        for d in [-1i64, 0, 1] {
            let x = b + d;
            v.push(x as i32);
            v.push(-x as i32);
        }
    }
    v.sort_unstable();
    v.dedup();
    v
}

#[inline]
fn recon_oracle(scale_q: i32, quantile_q: i32) -> i128 {
    (scale_q as i128 * quantile_q as i128) >> SCALE_SHIFT
}

#[test]
fn frozen_lut_entries_within_clamp() {
    let mut checked = 0usize;
    for l in FROZEN_MIN_L..=FROZEN_MAX_L {
        let q = quantile_lut(l);
        let c = codebook_lut(l);
        assert_eq!(q.len(), 1usize << l);
        assert_eq!(c.len(), 1usize << l);
        for &v in q.iter().chain(c.iter()) {
            assert!(
                (-Q_CLAMP..=Q_CLAMP).contains(&v),
                "LUT entry {v} outside ±{Q_CLAMP} at L={l}"
            );
            checked += 1;
        }
        
        let mut qs = q.to_vec();
        let mut cs = c.to_vec();
        qs.sort_unstable();
        cs.sort_unstable();
        assert_eq!(qs, cs, "codebook L={l} is not a permutation of the quantile LUT");
    }
    
    assert_eq!(checked, 65_504, "coverage accounting drifted");
}

#[test]
fn reconstruct_q_total_corners() {
    for s in [i32::MIN, i32::MIN + 1, -1, 0, 1, i32::MAX - 1, i32::MAX] {
        for q in [-Q_CLAMP, -Q_CLAMP + 1, -1, 0, 1, Q_CLAMP - 1, Q_CLAMP] {
            let oracle = recon_oracle(s, q);
            
            let prod128 = s as i128 * q as i128;
            assert!(prod128.unsigned_abs() < 1u128 << 63, "i64 product overflow at ({s},{q})");
            
            assert!(
                oracle >= i32::MIN as i128 && oracle <= i32::MAX as i128,
                "result leaves i32 at ({s},{q}): {oracle}"
            );
            assert!(oracle.unsigned_abs() <= RECON_ABS_MAX as u128);
            
            assert_eq!(reconstruct_q(s, q) as i128, oracle, "impl != oracle at ({s},{q})");
        }
    }
}

#[test]
fn reconstruct_q_total_boundary_sweep() {
    let scales = boundary_scales();
    
    let mut qs: Vec<i32> = Vec::new();
    for l in FROZEN_MIN_L..=FROZEN_MAX_L {
        qs.extend_from_slice(quantile_lut(l));
    }
    qs.push(Q_CLAMP);
    qs.push(-Q_CLAMP);
    qs.sort_unstable();
    qs.dedup();
    let mut checked = 0u64;
    for &s in &scales {
        for &q in &qs {
            let oracle = recon_oracle(s, q);
            debug_assert!(oracle >= i32::MIN as i128 && oracle <= i32::MAX as i128);
            assert!(
                oracle >= i32::MIN as i128 && oracle <= i32::MAX as i128,
                "result leaves i32 at ({s},{q})"
            );
            assert_eq!(reconstruct_q(s, q) as i128, oracle, "impl != oracle at ({s},{q})");
            checked += 1;
        }
    }
    eprintln!(
        "reconstruct_q boundary sweep: {} scales x {} quantiles = {} pairs",
        scales.len(),
        qs.len(),
        checked
    );
}

#[test]
#[ignore = "heavy: 2^32 x 4 enumeration; run with --release -- --ignored"]
fn reconstruct_q_total_full_i32_at_extreme_q() {
    for q in [Q_CLAMP, -Q_CLAMP, Q_CLAMP - 1, -(Q_CLAMP - 1)] {
        let mut s = i32::MIN;
        loop {
            let oracle = recon_oracle(s, q);
            assert!(oracle >= i32::MIN as i128 && oracle <= i32::MAX as i128);
            assert_eq!(reconstruct_q(s, q) as i128, oracle);
            if s == i32::MAX {
                break;
            }
            s += 1;
        }
    }
}

#[test]
fn eff_scale_q_total() {
    let scales = boundary_scales();
    let mut checked = 0u64;
    for code in 0u16..=255 {
        let code = code as u8;
        let mult = (code as i128 & 0x3F) + 1;
        for &s in &scales {
            let prod = s as i128 * mult;
            assert!(prod.unsigned_abs() < 1u128 << 63);
            let oracle = prod >> 6;
            assert!(
                oracle >= i32::MIN as i128 && oracle <= i32::MAX as i128,
                "eff_scale_q leaves i32 at ({s},{code})"
            );
            assert_eq!(eff_scale_q(s, code) as i128, oracle, "impl != oracle at ({s},{code})");
            checked += 1;
        }
    }
    
    assert_eq!(eff_scale_q(i32::MIN, 63), i32::MIN);
    assert_eq!(eff_scale_q(i32::MAX, 63), i32::MAX);
    eprintln!("eff_scale_q sweep: 256 codes x {} scales = {checked} pairs", scales.len());
}

#[test]
fn eff_min_q_total_on_encoder_domain() {
    let bases: Vec<i32> = boundary_scales().into_iter().filter(|&b| b >= 0).collect();
    let mut checked = 0u64;
    for code in 0u16..=255 {
        let code = code as u8;
        let mag = (code & 0x1F) as i128;
        for &b in &bases {
            let oracle: i128 = if mag == 0 {
                0
            } else {
                let signed = if code & 0x20 != 0 { b as i128 * mag } else { -(b as i128 * mag) };
                signed / 31
            };
            assert!(
                oracle >= i32::MIN as i128 && oracle <= i32::MAX as i128,
                "eff_min_q leaves i32 at ({b},{code})"
            );
            assert_eq!(eff_min_q(b, code) as i128, oracle, "impl != oracle at ({b},{code})");
            
            assert!(oracle.unsigned_abs() <= b.unsigned_abs() as u128);
            checked += 1;
        }
    }
    eprintln!("eff_min_q sweep: 256 codes x {} bases = {checked} pairs", bases.len());
}

#[test]
fn eff_min_q_i32_min_wrap_is_out_of_domain() {
    assert_eq!(eff_min_q(i32::MIN, 0x3F), i32::MIN, "wrap behaviour changed — update the ledger");
    
    assert_eq!(eff_min_q(i32::MIN, 0x1F), i32::MIN);
}

#[test]
fn recon_plus_min_offset_add_bound() {
    let max_safe_base: i64 = i32::MAX as i64 - RECON_ABS_MAX; 
    assert_eq!(max_safe_base, 1_342_177_279);
    
    let recon_neg = reconstruct_q(i32::MIN, Q_CLAMP);
    assert_eq!(recon_neg as i64, -RECON_ABS_MAX); 
    let recon_pos = reconstruct_q(i32::MAX, Q_CLAMP);
    assert_eq!(recon_pos as i64, RECON_ABS_MAX - 1); 
    
    let tight = max_safe_base + 1;
    assert!(recon_pos as i64 + tight <= i32::MAX as i64);
    assert!(recon_neg as i64 - tight >= i32::MIN as i64);
    
    assert!(
        recon_pos as i64 + (tight + 1) > i32::MAX as i64,
        "the bound is not tight — recompute the ledger"
    );
    
    assert_eq!(eff_min_q(max_safe_base as i32, 0x3F) as i64, max_safe_base);
}

fn assert_reader_matches(bytes: &[u8], start: usize, k: u32, n_syms: usize) {
    let mut reader = WordBitReader::new(bytes, start);
    let mut cursor = start;
    for i in 0..n_syms {
        let want = read_bits(bytes, cursor, k);
        let got = reader.pop(k);
        assert_eq!(got, want, "k={k} start={start} sym#{i} bytes={bytes:02x?}");
        cursor += k as usize;
    }
}

#[test]
fn word_reader_exhaustive_two_byte_buffers() {
    for content in 0u32..=0xFFFF {
        let bytes = [(content & 0xFF) as u8, (content >> 8) as u8];
        for k in 1..=8u32 {
            let n_syms = 16 / k as usize + 9; 
            assert_reader_matches(&bytes, 0, k, n_syms);
        }
    }
}

#[test]
fn word_reader_exhaustive_two_byte_buffers_all_offsets() {
    for content in 0u32..=0xFFFF {
        let bytes = [(content & 0xFF) as u8, (content >> 8) as u8];
        for start in 0..16usize {
            for k in [1u32, 3, 8] {
                let n_syms = (16 - start) / k as usize + 6;
                assert_reader_matches(&bytes, start, k, n_syms);
            }
        }
    }
}

#[test]
fn word_reader_bit_basis_16_byte_buffers_full_grid() {
    let mut bufs: Vec<[u8; 16]> = vec![[0u8; 16]];
    for bit in 0..128usize {
        let mut b = [0u8; 16];
        b[bit / 8] = 1u8 << (bit % 8);
        bufs.push(b);
    }
    let mut checked = 0u64;
    for bytes in &bufs {
        for start in 0..64usize {
            for k in 1..=8u32 {
                let n_syms = (128 - start) / k as usize + 5;
                assert_reader_matches(bytes, start, k, n_syms);
                checked += 1;
            }
        }
    }
    eprintln!("word reader basis grid: {} buffers x 64 offsets x 8 k = {checked} drains", bufs.len());
}

#[test]
fn word_reader_xor_homomorphism() {
    let pattern = |seed: u32| -> [u8; 16] {
        let mut b = [0u8; 16];
        for (i, slot) in b.iter_mut().enumerate() {
            *slot = ((seed.wrapping_mul(2654435761).wrapping_add(i as u32 * 0x9E37))
                >> ((i % 4) * 7)) as u8;
        }
        b
    };
    for sa in 0..32u32 {
        for sb in 0..32u32 {
            let a = pattern(sa);
            let b = pattern(sb.wrapping_add(7777));
            let mut x = [0u8; 16];
            for i in 0..16 {
                x[i] = a[i] ^ b[i];
            }
            for k in [1u32, 3, 5, 8] {
                for start in [0usize, 1, 7, 13, 31, 63] {
                    let mut ra = WordBitReader::new(&a, start);
                    let mut rb = WordBitReader::new(&b, start);
                    let mut rx = WordBitReader::new(&x, start);
                    for _ in 0..(128 / k as usize + 2) {
                        assert_eq!(rx.pop(k), ra.pop(k) ^ rb.pop(k));
                    }
                }
            }
        }
    }
}

#[cfg(kani)]
mod kani_harnesses {
    use super::*;

    #[kani::proof]
    fn reconstruct_q_total_symbolic() {
        let s: i32 = kani::any();
        let q: i32 = kani::any();
        kani::assume((-Q_CLAMP..=Q_CLAMP).contains(&q));
        let r = reconstruct_q(s, q);
        let prod = (s as i64) * (q as i64); 
        assert_eq!(r as i64, prod >> SCALE_SHIFT);
    }

    #[kani::proof]
    fn eff_scale_q_total_symbolic() {
        let s: i32 = kani::any();
        let c: u8 = kani::any();
        let r = eff_scale_q(s, c);
        let mult = (c as i128 & 0x3F) + 1;
        assert_eq!(r as i128, (s as i128 * mult) >> 6);
    }

    #[kani::proof]
    fn eff_min_q_total_symbolic() {
        let b: i32 = kani::any();
        kani::assume(b >= 0);
        let c: u8 = kani::any();
        let r = eff_min_q(b, c);
        let mag = (c & 0x1F) as i128;
        let oracle: i128 = if mag == 0 {
            0
        } else if c & 0x20 != 0 {
            (b as i128 * mag) / 31
        } else {
            -(b as i128 * mag) / 31
        };
        assert_eq!(r as i128, oracle);
        assert!(r.unsigned_abs() <= b.unsigned_abs());
    }

    #[kani::proof]
    #[kani::unwind(12)]
    fn word_reader_matches_read_bits_symbolic() {
        let bytes: [u8; 6] = kani::any();
        let k: u32 = kani::any();
        kani::assume((1..=8).contains(&k));
        let start: usize = kani::any();
        kani::assume(start < 16);
        let mut reader = WordBitReader::new(&bytes, start);
        let mut cursor = start;
        let mut i = 0u32;
        while i < 8 {
            assert_eq!(reader.pop(k), read_bits(&bytes, cursor, k));
            cursor += k as usize;
            i += 1;
        }
    }
}

#[test]
fn word_reader_empty_buffer_reads_zero() {
    let bytes: [u8; 0] = [];
    for start in 0..64usize {
        for k in 1..=8u32 {
            let mut r = WordBitReader::new(&bytes, start);
            for i in 0..40 {
                assert_eq!(r.pop(k), 0, "k={k} start={start} sym#{i}");
                assert_eq!(read_bits(&bytes, start + i * k as usize, k), 0);
            }
        }
    }
}
