//! Direct-quant CPU operators. The matmul reads a quantized weight tensor VIEW and dequantizes one
//! output row's blocks into a bounded tile, dots it with the input, and moves on — the whole weight is
//! never expanded to f32. Numerics (f64 RMSNorm, NeoX RoPE, out-major f32 accumulation, f16-rounded
//! tied embedding) match Candidate B, so the greedy output is bit-identical. This is the CPU fallback
//! and the parity reference; Metal accelerates the same math.

use crate::gguf::GgmlType;
use crate::quant;
use crate::Result;

/// Out-major GEMV directly on a quantized weight view: out[r] = Σ_c dequant(W)[r,c] * x[c], f32 accum.
/// A single reusable `cols`-wide tile holds one dequantized row at a time.
pub fn gemv_quant(dtype: GgmlType, bytes: &[u8], rows: usize, cols: usize, x: &[f32], out: &mut [f32]) -> Result<()> {
    let mut tile = vec![0f32; cols];
    for r in 0..rows {
        quant::dequant_row(dtype, bytes, r, cols, &mut tile)?;
        let mut acc = 0f32;
        for c in 0..cols {
            acc += tile[c] * x[c];
        }
        out[r] = acc;
    }
    Ok(())
}

/// Tied LM head directly on the Q8_0 embedding view, with the f16 round-trip (matches Candidate B):
/// logits[v] = Σ_c f16(dequant(embd)[v,c]) * x[c].
pub fn logits_tied(dtype: GgmlType, bytes: &[u8], hidden: usize, vocab: usize, x: &[f32], out: &mut [f32]) -> Result<()> {
    let mut tile = vec![0f32; hidden];
    for v in 0..vocab {
        quant::embed_row_f16(dtype, bytes, v, hidden, &mut tile)?;
        let mut acc = 0f32;
        for c in 0..hidden {
            acc += tile[c] * x[c];
        }
        out[v] = acc;
    }
    Ok(())
}

/// Embedding lookup with the f16 round-trip (single row).
pub fn embed_lookup(dtype: GgmlType, bytes: &[u8], token: usize, hidden: usize, out: &mut [f32]) -> Result<()> {
    quant::embed_row_f16(dtype, bytes, token, hidden, out)
}

pub fn rmsnorm(x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
    let n = x.len();
    let mut sum_sq = 0.0f64;
    for &v in x {
        sum_sq += (v as f64) * (v as f64);
    }
    let inv = 1.0f32 / (((sum_sq / n as f64) as f32) + eps).sqrt();
    for i in 0..n {
        out[i] = x[i] * inv * weight[i];
    }
}

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

pub fn silu_mul(gate: &[f32], up: &[f32], out: &mut [f32]) {
    for i in 0..gate.len() {
        let g = gate[i];
        out[i] = (g / (1.0f32 + (-g).exp())) * up[i];
    }
}

pub fn add_inplace(dst: &mut [f32], add: &[f32]) {
    for i in 0..dst.len() {
        dst[i] += add[i];
    }
}

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
    use crate::quant::f32_to_f16_bits;

    #[test]
    fn gemv_quant_matches_dense_for_q8_0() {
        // 3-row x 32-col Q8_0 weight, d=1; direct-quant gemv must equal a dense reference.
        let (rows, cols) = (3usize, 32usize);
        let mut bytes = vec![0u8; rows * 34];
        let mut dense = vec![0f32; rows * cols];
        for r in 0..rows {
            let d = f32_to_f16_bits(1.0).to_le_bytes();
            bytes[r * 34] = d[0];
            bytes[r * 34 + 1] = d[1];
            for i in 0..32 {
                let q = ((r + i) % 7) as i8;
                bytes[r * 34 + 2 + i] = q as u8;
                dense[r * cols + i] = q as f32;
            }
        }
        let x: Vec<f32> = (0..cols).map(|c| (c as f32) * 0.01).collect();
        let mut out = vec![0f32; rows];
        gemv_quant(GgmlType::Q8_0, &bytes, rows, cols, &x, &mut out).unwrap();
        for r in 0..rows {
            let want: f32 = (0..cols).map(|c| dense[r * cols + c] * x[c]).sum();
            assert!((out[r] - want).abs() < 1e-5);
        }
    }
}
