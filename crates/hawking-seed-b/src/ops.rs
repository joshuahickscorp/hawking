//! The scalar CPU operations the forward pass needs, each matched to the predecessor's numerics for
//! bit-identical decode: RMSNorm (f64 sum-of-squares), NeoX RoPE, out-major f32 GEMV, stable softmax,
//! SiLU-mul, argmax. No SIMD, no BLAS — correctness first (the directive's requirement).

/// RMSNorm with f64 sum-of-squares, mean cast to f32 BEFORE adding eps, eps inside sqrt, and the
/// exact multiply order `x[i] * inv * weight[i]`.
pub fn rmsnorm(x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
    let n = x.len();
    let mut sum_sq = 0.0f64;
    for &v in x {
        sum_sq += (v as f64) * (v as f64);
    }
    let rms = ((sum_sq / n as f64) as f32 + eps).sqrt();
    let inv = 1.0f32 / rms;
    for i in 0..n {
        out[i] = x[i] * inv * weight[i];
    }
}

/// Out-major GEMV: `out[r] = Σ_c W[r*cols + c] * x[c]`, accumulated in f32. Each output row of the
/// weight is contiguous over `cols` (= in_features). rows = out_features.
pub fn gemv(w: &[f32], x: &[f32], rows: usize, cols: usize, out: &mut [f32]) {
    debug_assert_eq!(x.len(), cols);
    for r in 0..rows {
        let base = r * cols;
        let mut acc = 0.0f32;
        for c in 0..cols {
            acc += w[base + c] * x[c];
        }
        out[r] = acc;
    }
}

/// Out-major GEMV where the weight rows are f16 (bits), converted to f32 per element (tied LM head +
/// embedding-as-matrix). Accumulation in f32.
pub fn gemv_f16(w_bits: &[u16], x: &[f32], rows: usize, cols: usize, out: &mut [f32]) {
    debug_assert_eq!(x.len(), cols);
    for r in 0..rows {
        let base = r * cols;
        let mut acc = 0.0f32;
        for c in 0..cols {
            acc += crate::quant::f16_to_f32(w_bits[base + c]) * x[c];
        }
        out[r] = acc;
    }
}

/// NeoX RoPE applied per head, in place: dim `i` pairs with `i + head_dim/2`. theta = pos / base^(2i/d).
pub fn rope_neox(x: &mut [f32], n_heads: usize, head_dim: usize, base: f32, pos: usize) {
    let half = head_dim / 2;
    for h in 0..n_heads {
        let hb = h * head_dim;
        for i in 0..half {
            let theta = (pos as f32) / base.powf(2.0 * i as f32 / head_dim as f32);
            let (sin, cos) = theta.sin_cos();
            let x0 = x[hb + i];
            let x1 = x[hb + i + half];
            x[hb + i] = x0 * cos - x1 * sin;
            x[hb + i + half] = x0 * sin + x1 * cos;
        }
    }
}

/// Numerically-stable softmax in place (subtract max), f32.
pub fn softmax(scores: &mut [f32]) {
    let mut m = f32::NEG_INFINITY;
    for &s in scores.iter() {
        if s > m {
            m = s;
        }
    }
    let mut sum = 0.0f32;
    for s in scores.iter_mut() {
        *s = (*s - m).exp();
        sum += *s;
    }
    let inv = 1.0f32 / sum;
    for s in scores.iter_mut() {
        *s *= inv;
    }
}

/// SiLU(gate) * up, elementwise: silu(x) = x / (1 + exp(-x)).
pub fn silu_mul(gate: &[f32], up: &[f32], out: &mut [f32]) {
    for i in 0..gate.len() {
        let g = gate[i];
        let s = g / (1.0f32 + (-g).exp());
        out[i] = s * up[i];
    }
}

/// Elementwise residual add: `dst[i] += add[i]`.
pub fn add_inplace(dst: &mut [f32], add: &[f32]) {
    for i in 0..dst.len() {
        dst[i] += add[i];
    }
}

/// Greedy argmax with lowest-index tie-break (strict `>`, ascending scan).
pub fn argmax(xs: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut bv = f32::NEG_INFINITY;
    for (i, &v) in xs.iter().enumerate() {
        if v > bv {
            best = i;
            bv = v;
        }
    }
    best as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rmsnorm_unit_vector() {
        let x = [3.0f32, 4.0];
        let w = [1.0f32, 1.0];
        let mut out = [0.0f32; 2];
        rmsnorm(&x, &w, 0.0, &mut out);
        // rms = sqrt((9+16)/2) = sqrt(12.5); out = x / rms
        let rms = (12.5f32).sqrt();
        assert!((out[0] - 3.0 / rms).abs() < 1e-6);
        assert!((out[1] - 4.0 / rms).abs() < 1e-6);
    }

    #[test]
    fn gemv_identity() {
        // 2x2 identity
        let w = [1.0f32, 0.0, 0.0, 1.0];
        let x = [5.0f32, 7.0];
        let mut out = [0.0f32; 2];
        gemv(&w, &x, 2, 2, &mut out);
        assert_eq!(out, [5.0, 7.0]);
    }

    #[test]
    fn rope_pos0_is_identity() {
        let mut x = [1.0f32, 2.0, 3.0, 4.0];
        rope_neox(&mut x, 1, 4, 10000.0, 0);
        assert_eq!(x, [1.0, 2.0, 3.0, 4.0]); // pos 0 -> theta 0 -> cos1 sin0
    }

    #[test]
    fn softmax_and_argmax() {
        let mut s = [1.0f32, 2.0, 3.0];
        softmax(&mut s);
        let sum: f32 = s.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert_eq!(argmax(&[0.1, 0.9, 0.9]), 1); // tie -> lowest index
    }

    #[test]
    fn silu_mul_zero_gate() {
        let mut out = [0.0f32; 1];
        silu_mul(&[0.0], &[5.0], &mut out);
        assert_eq!(out[0], 0.0); // silu(0)=0
    }
}
