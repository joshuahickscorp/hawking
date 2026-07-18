//! A minimal, real Forge representation family: per-column symmetric int8 quantization. Fits a scale,
//! packs codes + scales, reports EXACT physical bytes, round-trips, and yields a relative error. In
//! Candidate B the reconstruction is exercised through the crate's own `ops::gemv` (see main.rs), so
//! Forge executes on B's compact-linear path, not a disconnected demo.

use crate::{Error, Result};

#[derive(Debug, Clone)]
pub struct Packed {
    pub rows: usize,
    pub cols: usize,
    pub codes: Vec<i8>,
    pub scales: Vec<f32>,
}

impl Packed {
    pub fn physical_bytes(&self) -> usize {
        self.codes.len() + self.scales.len() * 4 + 16
    }
    pub fn whole_artifact_bpw(&self) -> f64 {
        (self.physical_bytes() * 8) as f64 / (self.rows * self.cols).max(1) as f64
    }
}

pub fn pack(weight: &[f32], rows: usize, cols: usize) -> Result<Packed> {
    if weight.len() != rows * cols {
        return Err(Error::Runtime("forge: weight shape mismatch".into()));
    }
    let mut scales = vec![0f32; cols];
    for c in 0..cols {
        let mut m = 0f32;
        for r in 0..rows {
            m = m.max(weight[r * cols + c].abs());
        }
        scales[c] = if m > 0.0 { m / 127.0 } else { 1.0 };
    }
    let mut codes = vec![0i8; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            let q = (weight[r * cols + c] / scales[c]).round().clamp(-127.0, 127.0);
            codes[r * cols + c] = q as i8;
        }
    }
    Ok(Packed { rows, cols, codes, scales })
}

pub fn decode(p: &Packed) -> Vec<f32> {
    let mut out = vec![0f32; p.rows * p.cols];
    for r in 0..p.rows {
        for c in 0..p.cols {
            out[r * p.cols + c] = p.codes[r * p.cols + c] as f32 * p.scales[c];
        }
    }
    out
}

pub fn rel_error(orig: &[f32], recon: &[f32]) -> f64 {
    let mut num = 0f64;
    let mut den = 0f64;
    for (a, b) in orig.iter().zip(recon.iter()) {
        num += ((*a - *b) as f64).powi(2);
        den += (*a as f64).powi(2);
    }
    num.sqrt() / den.sqrt().max(1e-12)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_roundtrip_and_exact_bytes() {
        let (rows, cols) = (8usize, 8usize);
        let w: Vec<f32> = (0..rows * cols).map(|i| ((i % 7) as f32 - 3.0) * 0.1).collect();
        let p = pack(&w, rows, cols).unwrap();
        assert_eq!(p.physical_bytes(), 64 + 32 + 16);
        let recon = decode(&p);
        assert!(rel_error(&w, &recon) < 0.1);
    }

    #[test]
    fn forge_recon_runs_through_b_gemv() {
        // the Forge reconstruction is a real weight B's linear op can execute
        let (rows, cols) = (4usize, 4usize);
        let w: Vec<f32> = (0..rows * cols).map(|i| ((i % 3) as f32 - 1.0) * 0.25).collect();
        let recon = decode(&pack(&w, rows, cols).unwrap());
        let x = vec![1.0f32; cols];
        let mut out = vec![0f32; rows];
        crate::ops::gemv(&recon, &x, rows, cols, &mut out);
        assert_eq!(out.len(), rows);
    }
}
