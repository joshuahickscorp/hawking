//! A minimal, real Doctor treatment: observe the Forge reconstruction error, diagnose it, allocate a
//! sparse outlier-column correction INSIDE the same physical budget, apply it, re-evaluate, and return
//! evidence. Doctor bytes are counted in the total BPW (Gravity law).

use crate::forge::{self, Packed};
use crate::gravity;
use crate::{Error, Result};

#[derive(Debug, Clone)]
pub struct Treatment {
    pub diagnosis: String,
    pub corrected_cols: Vec<usize>,
    /// fp16 correction values for the worst columns (billed).
    pub correction: Vec<(usize, Vec<f32>)>,
    pub doctor_bytes: usize,
}

#[derive(Debug, Clone)]
pub struct TreatmentReport {
    pub before_rel_error: f64,
    pub after_rel_error: f64,
    pub base_bpw: f64,
    pub doctor_bpw: f64,
    pub total_bpw: f64,
    pub within_budget: bool,
    pub improved: bool,
}

/// Observe -> diagnose -> allocate (n_cols worst) -> apply -> re-evaluate, within `budget_bpw`.
pub fn treat(orig: &[f32], p: &Packed, n_cols: usize, budget_bpw: f64) -> Result<(Treatment, TreatmentReport)> {
    let recon = forge::decode(p);
    let before = forge::rel_error(orig, &recon);

    // diagnose: per-column reconstruction energy; pick the worst columns.
    let mut col_err: Vec<(usize, f64)> = (0..p.cols)
        .map(|c| {
            let e: f64 = (0..p.rows)
                .map(|r| {
                    let i = r * p.cols + c;
                    ((orig[i] - recon[i]) as f64).powi(2)
                })
                .sum();
            (c, e)
        })
        .collect();
    col_err.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let worst: Vec<usize> = col_err.iter().take(n_cols.min(p.cols)).map(|(c, _)| *c).collect();

    // apply: store the worst columns' original values in fp16 (billed); reconstruct exactly there.
    let mut corrected = recon.clone();
    let mut correction = Vec::new();
    for &c in &worst {
        let mut vals = Vec::with_capacity(p.rows);
        for r in 0..p.rows {
            let i = r * p.cols + c;
            corrected[i] = orig[i];
            vals.push(orig[i]);
        }
        correction.push((c, vals));
    }
    let after = forge::rel_error(orig, &corrected);

    // physical accounting: fp16 correction values + a column index each.
    let doctor_bytes = worst.len() * (p.rows * 2 + 4);
    let n = (p.rows * p.cols) as u64;
    let base_bpw = (p.physical_bytes() * 8) as f64 / n as f64;
    let doctor_bpw = (doctor_bytes * 8) as f64 / n as f64;
    let total = gravity::total_bpw(p.physical_bytes() as u64 * 8, doctor_bytes as u64 * 8, 0, n);
    let within = gravity::doctor_within_budget(
        p.physical_bytes() as u64 * 8,
        doctor_bytes as u64 * 8,
        0,
        budget_bpw,
        n,
    )
    .is_ok();

    if !within {
        return Err(Error::Gravity(format!(
            "Doctor treatment exceeds budget: {total:.3} > {budget_bpw:.3} BPW"
        )));
    }

    Ok((
        Treatment {
            diagnosis: format!("{} outlier columns dominate the reconstruction error", worst.len()),
            corrected_cols: worst,
            correction,
            doctor_bytes,
        },
        TreatmentReport {
            before_rel_error: before,
            after_rel_error: after,
            base_bpw,
            doctor_bpw,
            total_bpw: total,
            within_budget: within,
            improved: after < before,
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn treatment_improves_within_budget_and_conserves_bytes() {
        let (rows, cols) = (8usize, 8usize);
        // one column is a large outlier the int8 pack mangles
        let mut w: Vec<f32> = (0..rows * cols).map(|i| ((i % 5) as f32 - 2.0) * 0.05).collect();
        for r in 0..rows {
            w[r * cols + 3] = if r % 2 == 0 { 9.0 } else { -9.0 };
        }
        let p = forge::pack(&w, rows, cols).unwrap();
        let (t, rep) = treat(&w, &p, 2, 20.0).unwrap();
        assert!(rep.improved, "Doctor must lower the error");
        assert!(rep.within_budget, "Doctor bytes must fit the budget");
        assert!(rep.total_bpw >= rep.base_bpw + rep.doctor_bpw - 1e-9, "total counts doctor bytes");
        assert!(!t.corrected_cols.is_empty());
    }
}
