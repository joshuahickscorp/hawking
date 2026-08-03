
pub const HADAMARD_BLOCK: usize = 256;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RhtConfig {
    
    pub seed: u64,
    
    pub block: usize,
}

impl RhtConfig {
    
    pub fn from_seed(seed: u64) -> Self {
        RhtConfig {
            seed,
            block: HADAMARD_BLOCK,
        }
    }
}

#[inline]
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[inline]
fn sign_at(seed: u64, i: usize) -> f32 {
    
    let mut s = seed ^ (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    let z = splitmix64(&mut s);
    if (z >> 63) & 1 == 0 {
        1.0
    } else {
        -1.0
    }
}

fn fwht_inplace(x: &mut [f32]) {
    let n = x.len();
    debug_assert!(n.is_power_of_two(), "FWHT length must be a power of two");
    let mut len = 1usize;
    while len < n {
        let mut i = 0usize;
        while i < n {
            for j in i..i + len {
                let u = x[j];
                let v = x[j + len];
                x[j] = u + v;
                x[j + len] = u - v;
            }
            i += len * 2;
        }
        len *= 2;
    }
    let scale = 1.0f32 / (n as f32).sqrt();
    for xi in x.iter_mut() {
        *xi *= scale;
    }
}

#[inline]
fn pow2_block_for(len: usize, cap: usize) -> usize {
    if len == 0 {
        return 1;
    }
    
    let div = 1usize << len.trailing_zeros();
    div.min(cap.max(1)).max(1)
}

pub fn rht_forward_inplace(x: &mut [f32], cfg: &RhtConfig) {
    let seed = cfg.seed;
    
    for (i, xi) in x.iter_mut().enumerate() {
        *xi *= sign_at(seed, i);
    }
    
    let h = pow2_block_for(x.len(), cfg.block);
    let mut start = 0usize;
    let n = x.len();
    while start + h <= n {
        fwht_inplace(&mut x[start..start + h]);
        start += h;
    }
    
}

pub fn rht_inverse_inplace(x: &mut [f32], cfg: &RhtConfig) {
    let seed = cfg.seed;
    let h = pow2_block_for(x.len(), cfg.block);
    let mut start = 0usize;
    let n = x.len();
    while start + h <= n {
        fwht_inplace(&mut x[start..start + h]);
        start += h;
    }
    for (i, xi) in x.iter_mut().enumerate() {
        *xi *= sign_at(seed, i);
    }
}

pub fn rht_forward(x: &[f32], cfg: &RhtConfig) -> Vec<f32> {
    let mut v = x.to_vec();
    rht_forward_inplace(&mut v, cfg);
    v
}

pub fn rht_inverse(x: &[f32], cfg: &RhtConfig) -> Vec<f32> {
    let mut v = x.to_vec();
    rht_inverse_inplace(&mut v, cfg);
    v
}

pub fn rht_forward_rows_inplace(x: &mut [f32], cfg: &RhtConfig, in_features: usize) {
    if in_features == 0 || x.len() % in_features != 0 {
        rht_forward_inplace(x, cfg);
        return;
    }
    let seed = cfg.seed;
    
    for (i, xi) in x.iter_mut().enumerate() {
        *xi *= sign_at(seed, i);
    }
    
    let h = pow2_block_for(in_features, cfg.block);
    let mut base = 0usize;
    while base < x.len() {
        let mut start = 0usize;
        while start + h <= in_features {
            fwht_inplace(&mut x[base + start..base + start + h]);
            start += h;
        }
        base += in_features;
    }
}

pub fn rht_inverse_rows_inplace(x: &mut [f32], cfg: &RhtConfig, in_features: usize) {
    if in_features == 0 || x.len() % in_features != 0 {
        rht_inverse_inplace(x, cfg);
        return;
    }
    let seed = cfg.seed;
    let h = pow2_block_for(in_features, cfg.block);
    let mut base = 0usize;
    while base < x.len() {
        let mut start = 0usize;
        while start + h <= in_features {
            fwht_inplace(&mut x[base + start..base + start + h]);
            start += h;
        }
        base += in_features;
    }
    for (i, xi) in x.iter_mut().enumerate() {
        *xi *= sign_at(seed, i);
    }
}

pub fn rht_forward_rows(x: &[f32], cfg: &RhtConfig, in_features: usize) -> Vec<f32> {
    let mut v = x.to_vec();
    rht_forward_rows_inplace(&mut v, cfg, in_features);
    v
}

pub fn rht_inverse_rows(x: &[f32], cfg: &RhtConfig, in_features: usize) -> Vec<f32> {
    let mut v = x.to_vec();
    rht_inverse_rows_inplace(&mut v, cfg, in_features);
    v
}

// ---- per-COLUMN-sign RHT: the cheap-serving variant -------------------------------------
// Identical per-row FWHT to rht_forward_rows, but the random sign is shared across rows
// (sign_at(seed, col) instead of per-element sign_at(seed, row*in+col)). Consequence for
// inference: y = W·x = decode(W_rht)·T(x) where the activation transform T(x) = sign_at(seed,col)
// then FWHT is computed ONCE and reused for every output row — vs the per-row variant, which
// needs `out_features` distinct activation transforms per token (the ~1 tok/s serving wall).
// T(x) for a single in_features vector == rht_forward_inplace(x) (sign_at(seed,i=col) + FWHT).
pub fn rht_forward_cols_inplace(x: &mut [f32], cfg: &RhtConfig, in_features: usize) {
    if in_features == 0 || x.len() % in_features != 0 {
        rht_forward_inplace(x, cfg);
        return;
    }
    let seed = cfg.seed;
    for (i, xi) in x.iter_mut().enumerate() {
        *xi *= sign_at(seed, i % in_features);
    }
    let h = pow2_block_for(in_features, cfg.block);
    let mut base = 0usize;
    while base < x.len() {
        let mut start = 0usize;
        while start + h <= in_features {
            fwht_inplace(&mut x[base + start..base + start + h]);
            start += h;
        }
        base += in_features;
    }
}

pub fn rht_inverse_cols_inplace(x: &mut [f32], cfg: &RhtConfig, in_features: usize) {
    if in_features == 0 || x.len() % in_features != 0 {
        rht_inverse_inplace(x, cfg);
        return;
    }
    let seed = cfg.seed;
    let h = pow2_block_for(in_features, cfg.block);
    let mut base = 0usize;
    while base < x.len() {
        let mut start = 0usize;
        while start + h <= in_features {
            fwht_inplace(&mut x[base + start..base + start + h]);
            start += h;
        }
        base += in_features;
    }
    for (i, xi) in x.iter_mut().enumerate() {
        *xi *= sign_at(seed, i % in_features);
    }
}

pub fn rht_forward_cols(x: &[f32], cfg: &RhtConfig, in_features: usize) -> Vec<f32> {
    let mut v = x.to_vec();
    rht_forward_cols_inplace(&mut v, cfg, in_features);
    v
}

pub fn rht_inverse_cols(x: &[f32], cfg: &RhtConfig, in_features: usize) -> Vec<f32> {
    let mut v = x.to_vec();
    rht_inverse_cols_inplace(&mut v, cfg, in_features);
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    #[test]
    fn fwht_is_self_inverse() {
        let mut x: Vec<f32> = (0..256).map(|i| ((i as f32) * 0.7).sin()).collect();
        let orig = x.clone();
        fwht_inplace(&mut x);
        fwht_inplace(&mut x);
        assert!(max_abs_diff(&orig, &x) < 1e-4, "FWHT not involutive");
    }

    #[test]
    fn rht_round_trip_identity_pow2_multiple() {
        
        let n = 4864usize;
        let x: Vec<f32> = (0..n)
            .map(|i| ((i as f32) * 0.013).sin() * 0.3 + ((i % 7) as f32) * 0.05)
            .collect();
        let cfg = RhtConfig::from_seed(0xDEAD_BEEF_1234_5678);
        let fwd = rht_forward(&x, &cfg);
        let back = rht_inverse(&fwd, &cfg);
        let d = max_abs_diff(&x, &back);
        assert!(d < 1e-3, "RHT round-trip not identity: max abs diff {d}");
    }

    fn l2(v: &[f32]) -> f64 {
        v.iter().map(|&z| (z as f64) * (z as f64)).sum()
    }

    #[test]
    fn rht_round_trip_with_tail() {
        
        let n = 256 * 5 + 37;
        let x: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.021).cos()).collect();
        let cfg = RhtConfig::from_seed(0x0BAD_F00D);
        let back = rht_inverse(&rht_forward(&x, &cfg), &cfg);
        assert!(max_abs_diff(&x, &back) < 1e-3, "RHT+tail round-trip failed");
    }

    #[test]
    fn rht_rows_round_trip_awkward_in_features() {
        
        let cfg = RhtConfig::from_seed(0xA5A5_1234_DEAD_0001);
        for &(out_f, in_f) in &[(8usize, 896usize), (3, 896), (7, 100), (1, 100), (5, 768 + 128)] {
            let n = out_f * in_f;
            let x: Vec<f32> = (0..n)
                .map(|i| ((i as f32) * 0.013).sin() * 0.3 + ((i % 7) as f32) * 0.05)
                .collect();
            let fwd = rht_forward_rows(&x, &cfg, in_f);
            let back = rht_inverse_rows(&fwd, &cfg, in_f);
            let d = max_abs_diff(&x, &back);
            assert!(
                d < 1e-3,
                "row-aware RHT round-trip not identity (out={out_f}, in={in_f}): max abs diff {d}"
            );
            let (ex, ey) = (l2(&x), l2(&fwd));
            assert!(
                (ex - ey).abs() / ex.max(1e-12) < 1e-3,
                "row-aware RHT not energy-preserving (out={out_f}, in={in_f}): {ex} vs {ey}"
            );
        }
    }

    #[test]
    fn rht_rows_matches_flat_when_in_features_256_aligned() {
        
        let cfg = RhtConfig::from_seed(0xBEEF_F00D_0000_2222);
        for &(out_f, in_f) in &[(8usize, 256usize), (4, 512), (6, 768), (3, 1024), (5, 4864)] {
            assert_eq!(in_f % HADAMARD_BLOCK, 0, "test dim must be 256-aligned");
            let n = out_f * in_f;
            let x: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.017).cos() * 0.4).collect();
            let flat = rht_forward(&x, &cfg);
            let rows = rht_forward_rows(&x, &cfg, in_f);
            
            assert_eq!(
                flat, rows,
                "row-aware RHT diverged from flat at aligned in_features={in_f}"
            );
        }
    }

    #[test]
    fn rht_is_energy_preserving() {
        
        let n = 1024usize;
        let x: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.1).sin() + 0.2).collect();
        let cfg = RhtConfig::from_seed(42);
        let y = rht_forward(&x, &cfg);
        let ex: f64 = x.iter().map(|&v| (v as f64) * (v as f64)).sum();
        let ey: f64 = y.iter().map(|&v| (v as f64) * (v as f64)).sum();
        assert!(
            (ex - ey).abs() / ex.max(1e-12) < 1e-3,
            "RHT not energy-preserving: {ex} vs {ey}"
        );
    }

    #[test]
    fn rht_gaussianizes_clustered_weights() {
        
        let n = 8192usize;
        let x: Vec<f32> = (0..n)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 } + ((i % 13) as f32) * 0.01)
            .collect();
        let cfg = RhtConfig::from_seed(7);
        let y = rht_forward(&x, &cfg);
        let kurt = |v: &[f32]| -> f64 {
            let m = v.iter().map(|&z| z as f64).sum::<f64>() / v.len() as f64;
            let var = v.iter().map(|&z| (z as f64 - m).powi(2)).sum::<f64>() / v.len() as f64;
            let m4 = v.iter().map(|&z| (z as f64 - m).powi(4)).sum::<f64>() / v.len() as f64;
            m4 / (var * var)
        };
        let k_before = kurt(&x);
        let k_after = kurt(&y);
        
        assert!(
            (k_after - 3.0).abs() < (k_before - 3.0).abs(),
            "RHT did not move kurtosis toward Gaussian: before {k_before:.3} after {k_after:.3}"
        );
    }

    #[test]
    fn signs_are_deterministic() {
        
        for i in [0usize, 1, 100, 4863, 99999] {
            assert_eq!(sign_at(0xABCD, i), sign_at(0xABCD, i));
        }
        
        let signs: Vec<f32> = (0..1000).map(|i| sign_at(123, i)).collect();
        assert!(signs.iter().any(|&s| s > 0.0) && signs.iter().any(|&s| s < 0.0));
    }
}
