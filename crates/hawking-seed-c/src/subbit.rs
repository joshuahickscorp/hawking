//! A genuine sub-bit representation and a compact operator that executes it DIRECTLY (never expanding
//! to a dense matrix). Family: ternary latent factorization W ≈ scale·(A·B) where A (m×r) and B (r×n)
//! are ternary {-1,0,1}, packed 5 trits/byte (log2(3)≈1.6 bits/entry). For r ≪ mn/(m+n) the whole
//! physical BPW is well under 1.0. The operator computes y = scale·A·(B·x) with two ternary mat-vecs.
//! A Doctor rescue adds a sparse full-precision residual, executed directly, still inside the budget.

/// Ternary matrix packed 5 trits/byte, row-major, `rows`×`cols`.
#[derive(Debug, Clone)]
pub struct Ternary {
    pub rows: usize,
    pub cols: usize,
    pub packed: Vec<u8>, // ceil(rows*cols/5) bytes
}
impl Ternary {
    fn from_signs(signs: &[i8], rows: usize, cols: usize) -> Self {
        let n = rows * cols;
        let mut packed = vec![0u8; n.div_ceil(5)];
        for (i, &s) in signs.iter().enumerate() {
            let t = (s + 1) as u32; // {-1,0,1} -> {0,1,2}
            packed[i / 5] += (t * pow3(i % 5)) as u8;
        }
        Ternary { rows, cols, packed }
    }
    #[inline]
    fn get(&self, i: usize) -> i32 {
        let byte = self.packed[i / 5] as u32;
        let t = (byte / pow3(i % 5)) % 3;
        t as i32 - 1
    }
    pub fn bits(&self) -> usize {
        self.packed.len() * 8
    }
}
#[inline]
fn pow3(k: usize) -> u32 {
    [1, 3, 9, 27, 81][k]
}

/// A sparse full-precision Doctor correction: (row, col, f16 value) triples, applied directly.
#[derive(Debug, Clone, Default)]
pub struct Correction {
    pub entries: Vec<(usize, usize, f32)>,
}
impl Correction {
    pub fn bits(&self) -> usize {
        // f16 value (16) + row index (16) + col index (16) per entry
        self.entries.len() * 48
    }
}

pub struct SubBitMatrix {
    pub m: usize,
    pub n: usize,
    pub r: usize,
    pub a: Ternary, // m×r
    pub b: Ternary, // r×n
    pub scale: f32,
    pub correction: Correction,
}

impl SubBitMatrix {
    /// Physical bits = A trits + B trits + one f32 scale + Doctor correction.
    pub fn bits(&self) -> usize {
        self.a.bits() + self.b.bits() + 32 + self.correction.bits()
    }
    pub fn whole_bpw(&self) -> f64 {
        self.bits() as f64 / (self.m * self.n) as f64
    }

    /// Direct execution: y = scale·A·(B·x) + Σ correction, WITHOUT forming A·B.
    pub fn matvec(&self, x: &[f32]) -> Vec<f32> {
        // t = B·x  (r-vector), ternary mat-vec
        let mut t = vec![0f32; self.r];
        for i in 0..self.r {
            let mut acc = 0f32;
            for j in 0..self.n {
                let s = self.b.get(i * self.n + j);
                if s != 0 {
                    acc += s as f32 * x[j];
                }
            }
            t[i] = acc;
        }
        // y = scale·A·t  (m-vector)
        let mut y = vec![0f32; self.m];
        for i in 0..self.m {
            let mut acc = 0f32;
            for k in 0..self.r {
                let s = self.a.get(i * self.r + k);
                if s != 0 {
                    acc += s as f32 * t[k];
                }
            }
            y[i] = acc * self.scale;
        }
        // Doctor correction: y[row] += val · x[col]  (direct, sparse)
        for &(row, col, val) in &self.correction.entries {
            y[row] += val * x[col];
        }
        y
    }
}

/// Fit a ternary latent factorization of `w` (m×n, row-major) with latent rank `r`. Power iteration
/// finds a rank-r subspace; entries are ternarized; a global scale best-fits Frobenius.
pub fn fit(w: &[f32], m: usize, n: usize, r: usize) -> SubBitMatrix {
    // deterministic init of B (r×n) with a fixed pattern, then a few power iterations.
    let mut bf = vec![0f32; r * n];
    for i in 0..r {
        for j in 0..n {
            // deterministic pseudo-random in [-1,1]
            let s = (((i * 1103515245 + j * 12345 + 1013904223) >> 8) & 0xFFFF) as f32 / 32768.0 - 1.0;
            bf[i * n + j] = s;
        }
    }
    let mut af = vec![0f32; m * r];
    for _ in 0..4 {
        // A = W · Bᵀ  (m×r)
        for i in 0..m {
            for k in 0..r {
                let mut acc = 0f32;
                for j in 0..n {
                    acc += w[i * n + j] * bf[k * n + j];
                }
                af[i * r + k] = acc;
            }
        }
        // B = Aᵀ · W  (r×n), then row-normalize
        for k in 0..r {
            for j in 0..n {
                let mut acc = 0f32;
                for i in 0..m {
                    acc += af[i * r + k] * w[i * n + j];
                }
                bf[k * n + j] = acc;
            }
            let norm = (bf[k * n..(k + 1) * n].iter().map(|v| v * v).sum::<f32>()).sqrt().max(1e-8);
            for j in 0..n {
                bf[k * n + j] /= norm;
            }
        }
    }
    // ternarize (sign) A and B
    let a_signs: Vec<i8> = af.iter().map(|&v| tern(v)).collect();
    let b_signs: Vec<i8> = bf.iter().map(|&v| tern(v)).collect();
    let a = Ternary::from_signs(&a_signs, m, r);
    let b = Ternary::from_signs(&b_signs, r, n);

    // global scale: <W, AB> / <AB, AB>  (AB formed only here, for fitting — never at execution)
    let mut ab = vec![0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0f32;
            for k in 0..r {
                acc += a.get(i * r + k) as f32 * b.get(k * n + j) as f32;
            }
            ab[i * n + j] = acc;
        }
    }
    let mut num = 0f32;
    let mut den = 0f32;
    for i in 0..m * n {
        num += w[i] * ab[i];
        den += ab[i] * ab[i];
    }
    let scale = if den > 0.0 { num / den } else { 0.0 };
    SubBitMatrix { m, n, r, a, b, scale, correction: Correction::default() }
}

#[inline]
fn tern(v: f32) -> i8 {
    if v > 1e-6 {
        1
    } else if v < -1e-6 {
        -1
    } else {
        0
    }
}

/// Relative Frobenius error of the operator vs the dense reference on a probe input set.
pub fn output_divergence(w: &[f32], m: usize, n: usize, sb: &SubBitMatrix, xs: &[Vec<f32>]) -> f64 {
    let mut num = 0f64;
    let mut den = 0f64;
    for x in xs {
        let approx = sb.matvec(x);
        for i in 0..m {
            let mut exact = 0f32;
            for j in 0..n {
                exact += w[i * n + j] * x[j];
            }
            num += ((exact - approx[i]) as f64).powi(2);
            den += (exact as f64).powi(2);
        }
    }
    (num.sqrt()) / (den.sqrt().max(1e-12))
}

/// Doctor rescue: allocate `k` sparse full-precision residual entries (largest |W - scale·AB| entries),
/// executed directly, staying inside `budget_bpw`. Returns the treated matrix or None if over budget.
pub fn doctor_rescue(w: &[f32], m: usize, n: usize, mut sb: SubBitMatrix, k: usize, budget_bpw: f64) -> Option<SubBitMatrix> {
    // residual R = W - scale·AB (formed only for diagnosis, not execution)
    let mut resid: Vec<(f32, usize, usize)> = Vec::with_capacity(m * n);
    for i in 0..m {
        for j in 0..n {
            let mut ab = 0f32;
            for kk in 0..sb.r {
                ab += sb.a.get(i * sb.r + kk) as f32 * sb.b.get(kk * n + j) as f32;
            }
            let e = w[i * n + j] - sb.scale * ab;
            resid.push((e.abs(), i, j));
        }
    }
    resid.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    let mut corr = Correction::default();
    for &(_, i, j) in resid.iter().take(k) {
        let mut ab = 0f32;
        for kk in 0..sb.r {
            ab += sb.a.get(i * sb.r + kk) as f32 * sb.b.get(kk * n + j) as f32;
        }
        let val = w[i * n + j] - sb.scale * ab; // exact residual for this entry
        corr.entries.push((i, j, half::f16::from_f32(val).to_f32()));
    }
    sb.correction = corr;
    if sb.whole_bpw() <= budget_bpw {
        Some(sb)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn probe(n: usize, seed: usize) -> Vec<f32> {
        (0..n).map(|j| (((j * 2654435761 + seed) >> 7) & 0xFF) as f32 / 128.0 - 1.0).collect()
    }

    #[test]
    fn subbit_is_under_one_bpw_and_executes_directly() {
        let (m, n, r) = (256usize, 256usize, 32usize);
        let w: Vec<f32> = (0..m * n).map(|i| (((i * 48271) % 997) as f32 / 997.0 - 0.5) * 0.1).collect();
        let sb = fit(&w, m, n, r);
        assert!(sb.whole_bpw() < 1.0, "must be sub-bit, got {}", sb.whole_bpw());
        // executes without forming a dense matrix
        let y = sb.matvec(&probe(n, 1));
        assert_eq!(y.len(), m);
    }

    #[test]
    fn doctor_rescue_reduces_divergence_within_budget() {
        let (m, n, r) = (256usize, 256usize, 32usize);
        let w: Vec<f32> = (0..m * n).map(|i| (((i * 48271) % 997) as f32 / 997.0 - 0.5) * 0.1).collect();
        let sb = fit(&w, m, n, r);
        let xs: Vec<Vec<f32>> = (0..4).map(|s| probe(n, s)).collect();
        let before = output_divergence(&w, m, n, &sb, &xs);
        let treated = doctor_rescue(&w, m, n, sb, 400, 0.99).expect("within budget");
        let after = output_divergence(&w, m, n, &treated, &xs);
        assert!(treated.whole_bpw() <= 0.99, "still sub-bit after Doctor: {}", treated.whole_bpw());
        assert!(after < before, "Doctor must reduce divergence: {before} -> {after}");
    }
}
