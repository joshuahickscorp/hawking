//! Pure-Rust Eagle6 head forward pass.
//!
//! Mirrors `Eagle5Head.forward` in `colab/eagle5_train_pytorch.py`:
//!
//! 1. Look up prev_token embedding from `_token_embd` (stored as
//!    `[hidden, vocab]` row-major; we read the strided column `prev_tok`).
//! 2. Concatenate `[prev_embed | residual_in | intermediate]` →
//!    `(3 * hidden,)`.
//! 3. `in_proj` matmul → `(hidden,)`.
//! 4. For each block (1 or 2 depending on head config):
//!    - RMSNorm with `attn_norm`
//!    - Multi-head self-attention (S=1 → degenerate, attn_out = `out_proj(v_proj(h))`)
//!    - Residual add
//!    - RMSNorm with `mlp_norm`
//!    - SwiGLU: `down(silu(gate(h)) * up(h))`
//!    - Residual add
//! 5. `baseline = RMSNorm(residual_in, _output_norm)`
//! 6. `draft_hidden = baseline + residual_gate * x`
//! 7. `logits = draft_hidden @ _lm_head` → `(vocab,)`
//!
//! **S=1 only.** The runtime always invokes the head one token at a
//! time (auto-regressive draft chain), so we specialize for S=1.
//! Multi-position support is irrelevant at decode and would only
//! complicate the parity test. At S=1 the attention reduces to
//! `out_proj(v_proj(rmsnorm(x)))` because:
//! - `q · kᵀ / sqrt(d)` is a single scalar per head, plus diagonal
//!   mask `(eye(1)-1)*1e9 = 0`
//! - `softmax([x]) = [1.0]` exact (no fp drift, single element)
//! - multiplying `v` by 1.0 returns `v`
//!
//! So skipping q_proj + k_proj at runtime is bit-identical to the
//! PyTorch reference. We DO compute them for parity-test clarity
//! (and the runtime cost is small — q+k are 16.8M FMAs each on a
//! 2048-dim head, total ~33M, vs the 75M for the MLP). Optimization
//! is Phase A.3+.
//!
//! All compute is fp32 to match the PyTorch reference exactly. The
//! frozen `_token_embd` and `_lm_head` are stored f16; they're
//! converted to f32 lazily during the relevant matmul. Trainable
//! weights are stored f32.

use crate::speculate::eagle5::{TrainedBlock, TrainedConfig};
use half::f16;

/// RMS epsilon, matches `RMS_EPS = 1e-6` in eagle5_train_pytorch.py:71.
const RMS_EPS: f32 = 1e-6;

/// Run the Eagle6 head forward pass on a single token position (S=1).
///
/// Inputs:
/// - `prev_token`: the previous token id (last verified or previous draft)
/// - `residual_in`: residual stream at the verifier's capture layer (hidden,)
/// - `intermediate`: intermediate stream at the verifier's capture layer (hidden,)
/// - `config`, `in_proj`, `blocks`, `residual_gate`, `output_norm`,
///    `token_embd_f16`, `lm_head_f16`: the trained head's weights
///
/// Returns: vocab-length f32 logits.
///
/// **The caller must validate `prev_token < vocab_size`.** This function
/// does saturating clamp for safety but a real out-of-range value
/// indicates a bug upstream.
#[allow(clippy::too_many_arguments)]
pub fn forward_single_step(
    config: &TrainedConfig,
    in_proj: &[f32],
    blocks: &[TrainedBlock],
    residual_gate: f32,
    output_norm: &[f32],
    token_embd_f16: &[f16],
    lm_head_f16: &[f16],
    prev_token: u32,
    residual_in: &[f32],
    intermediate: &[f32],
) -> Vec<f32> {
    let h = config.hidden_dim;
    let v = config.vocab_size;
    let ff = config.ff_dim;
    debug_assert_eq!(residual_in.len(), h);
    debug_assert_eq!(intermediate.len(), h);
    debug_assert_eq!(output_norm.len(), h);
    debug_assert_eq!(in_proj.len(), h * 3 * h);
    debug_assert_eq!(token_embd_f16.len(), h * v);
    debug_assert_eq!(lm_head_f16.len(), h * v);
    debug_assert_eq!(blocks.len(), config.num_blocks);

    // (1) prev_embed = embed_table[prev_token] where embed_table is
    // (vocab, hidden). Storage is [hidden, vocab] so we read the
    // strided column at offset `prev_token` (one f16 per row).
    let prev = (prev_token as usize).min(v.saturating_sub(1));
    let mut prev_embed = vec![0.0f32; h];
    for i in 0..h {
        prev_embed[i] = token_embd_f16[i * v + prev].to_f32();
    }

    // (2) Concatenate [prev_embed | residual_in | intermediate] → (3h,).
    let mut concat = Vec::with_capacity(3 * h);
    concat.extend_from_slice(&prev_embed);
    concat.extend_from_slice(residual_in);
    concat.extend_from_slice(intermediate);

    // (3) in_proj matmul: x = in_proj @ concat   (PyTorch nn.Linear:
    //     weight [out, in], y = x @ W.T → y[j] = sum_i W[j,i] * x[i]).
    let mut x = matmul_no_bias(in_proj, &concat, h, 3 * h);

    // (4) Apply each transformer block.
    let n_heads = config.n_heads;
    for blk in blocks {
        // Pre-attn RMSNorm.
        let h_norm = rms_norm(&x, &blk.attn_norm, RMS_EPS);
        // Self-attention at S=1 (degenerate; see module-level docstring).
        let attn_out = attention_s1(&h_norm, blk, n_heads, h);
        // Residual add.
        for i in 0..h {
            x[i] += attn_out[i];
        }
        // Pre-mlp RMSNorm.
        let h_norm = rms_norm(&x, &blk.mlp_norm, RMS_EPS);
        // SwiGLU: down(silu(gate(h)) * up(h)).
        let mlp_out = swiglu(&h_norm, &blk.mlp_gate, &blk.mlp_up, &blk.mlp_down, h, ff);
        // Residual add.
        for i in 0..h {
            x[i] += mlp_out[i];
        }
    }

    // (5) baseline = RMSNorm(residual_in, _output_norm).
    let baseline = rms_norm(residual_in, output_norm, RMS_EPS);

    // (6) draft_hidden = baseline + residual_gate * x.
    let mut draft_hidden = vec![0.0f32; h];
    for i in 0..h {
        draft_hidden[i] = baseline[i] + residual_gate * x[i];
    }

    // (7) logits = draft_hidden @ _lm_head. _lm_head is stored [h, v]
    // (so column v is the row of weights that produces logit v):
    //   logits[k] = sum_i draft_hidden[i] * lm_head[i, k]
    //             = sum_i draft_hidden[i] * lm_head_f16[i*v + k]
    //
    // Parallel over hidden axis. Each thread accumulates a partial
    // vocab-length vector from its slice of the hidden axis; the partial
    // vectors are summed at the end. Cache pattern within each thread
    // remains sequential (inner loop walks contiguous row of lm_head),
    // so we get linear scaling without thrashing the LM-head bytes.
    //
    // Why parallelize *only* the LM head: it's 311M FMAs at the q3b shape
    // (hidden=2048 × vocab=151936) — dominates the head's compute. The
    // earlier transformer-block matmuls (in_proj, q/k/v/out, mlp) are
    // collectively only ~155M FMAs and are not the bottleneck. Threading
    // them too would add overhead with marginal benefit.
    //
    // FP32 sum-order changes slightly vs the single-threaded loop above
    // (partials sum is in thread-index order). Differences land in
    // ~1-2 ULP per logit which is invisible at our parity gate
    // (atol=5e-2 vs measured 3.5e-4 single-threaded — plenty of room).
    let n_threads = std::thread::available_parallelism()
        .map(|n| n.get().clamp(1, 8))
        .unwrap_or(4);
    let logits = if n_threads <= 1 || h < n_threads * 32 {
        // Below this threshold the spawn/join overhead exceeds the
        // parallelism win — fall through to the simple single-thread
        // path. (This branch is what unit tests exercise too.)
        let mut out = vec![0.0f32; v];
        for i in 0..h {
            let dh_i = draft_hidden[i];
            let row = &lm_head_f16[i * v..(i + 1) * v];
            for k in 0..v {
                out[k] += dh_i * row[k].to_f32();
            }
        }
        out
    } else {
        let chunk = h.div_ceil(n_threads);
        let partials: Vec<Vec<f32>> = std::thread::scope(|s| {
            let mut handles = Vec::with_capacity(n_threads);
            for t in 0..n_threads {
                let i0 = t * chunk;
                let i1 = ((t + 1) * chunk).min(h);
                if i0 >= i1 {
                    continue;
                }
                let dh = &draft_hidden;
                let lm = lm_head_f16;
                handles.push(s.spawn(move || {
                    let mut local = vec![0.0f32; v];
                    for i in i0..i1 {
                        let dh_i = dh[i];
                        let row = &lm[i * v..(i + 1) * v];
                        for k in 0..v {
                            local[k] += dh_i * row[k].to_f32();
                        }
                    }
                    local
                }));
            }
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        // Reduce partials → final logits. Reused first buffer to avoid
        // an extra alloc.
        let mut iter = partials.into_iter();
        let mut out = iter.next().unwrap_or_else(|| vec![0.0f32; v]);
        for part in iter {
            for k in 0..v {
                out[k] += part[k];
            }
        }
        out
    };
    logits
}

/// RMSNorm matching `_rms_norm` in eagle5_train_pytorch.py:79-84:
///   x * w / sqrt(mean(x^2) + eps)
/// fp32 internally.
fn rms_norm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    debug_assert_eq!(weight.len(), n);
    let mut ss = 0.0_f64;
    for &xi in x.iter() {
        ss += (xi as f64) * (xi as f64);
    }
    // mean(x^2) = ss / n. rsqrt(mean + eps).
    let mean = ss / (n as f64);
    let rms = (mean + eps as f64).sqrt();
    let scale = 1.0 / rms as f32;
    let mut out = vec![0.0f32; n];
    for i in 0..n {
        out[i] = x[i] * scale * weight[i];
    }
    out
}

/// f16-weight version of matmul_no_bias for the verifier's LM head.
///
/// `y[j] = sum_i w_f16[j * in_dim + i].to_f32() * x[i]` for `j in 0..out_dim`.
///
/// Used by `forward_tokens_batched_with_logits` in qwen_dense.rs to do
/// per-position CPU LM-head dispatch after the GPU layer-stack runs.
/// Same threading pattern as `matmul_no_bias` (per-row independence,
/// bit-identical to single-threaded). Reusing from the Eagle5 module
/// keeps the threaded-matmul logic in one place.
pub fn matmul_no_bias_f16w(
    w_f16: &[f16],
    x: &[f32],
    out_dim: usize,
    in_dim: usize,
) -> Vec<f32> {
    debug_assert_eq!(w_f16.len(), out_dim * in_dim);
    debug_assert_eq!(x.len(), in_dim);

    let n_threads = std::thread::available_parallelism()
        .map(|n| n.get().clamp(1, 8))
        .unwrap_or(4);
    let min_rows_per_thread = 64;
    if n_threads <= 1 || out_dim < n_threads * min_rows_per_thread {
        let mut y = vec![0.0f32; out_dim];
        for j in 0..out_dim {
            let row = &w_f16[j * in_dim..(j + 1) * in_dim];
            let mut acc = 0.0_f32;
            for i in 0..in_dim {
                acc += row[i].to_f32() * x[i];
            }
            y[j] = acc;
        }
        return y;
    }

    let mut y = vec![0.0f32; out_dim];
    let chunk = out_dim.div_ceil(n_threads);
    std::thread::scope(|s| {
        let mut handles = Vec::with_capacity(n_threads);
        for (t, y_chunk) in y.chunks_mut(chunk).enumerate() {
            let j0 = t * chunk;
            let w_ref = w_f16;
            let x_ref = x;
            handles.push(s.spawn(move || {
                for (local_j, slot) in y_chunk.iter_mut().enumerate() {
                    let j = j0 + local_j;
                    let row = &w_ref[j * in_dim..(j + 1) * in_dim];
                    let mut acc = 0.0_f32;
                    for i in 0..in_dim {
                        acc += row[i].to_f32() * x_ref[i];
                    }
                    *slot = acc;
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
    });
    y
}

/// Matmul `y = W @ x` where `W` is row-major `[out_dim, in_dim]`. Matches
/// PyTorch nn.Linear(in, out) without bias: y[j] = sum_i W[j, i] * x[i].
///
/// Threaded over the output axis: each thread computes a contiguous range
/// of output rows. Because each output row's inner sum is independent
/// (no cross-thread accumulation), the threaded result is BIT-FOR-BIT
/// identical to the single-threaded loop — no FP sum-order drift to
/// worry about. Compare to the LM-head matmul in `forward_single_step`
/// which uses split-by-hidden + partial reduce (different sum order, but
/// well within the parity gate).
fn matmul_no_bias(w: &[f32], x: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
    debug_assert_eq!(w.len(), out_dim * in_dim);
    debug_assert_eq!(x.len(), in_dim);

    let n_threads = std::thread::available_parallelism()
        .map(|n| n.get().clamp(1, 8))
        .unwrap_or(4);
    // Below this threshold the spawn/join overhead exceeds the win.
    let min_rows_per_thread = 64;
    if n_threads <= 1 || out_dim < n_threads * min_rows_per_thread {
        let mut y = vec![0.0f32; out_dim];
        for j in 0..out_dim {
            let row = &w[j * in_dim..(j + 1) * in_dim];
            let mut acc = 0.0_f32;
            for i in 0..in_dim {
                acc += row[i] * x[i];
            }
            y[j] = acc;
        }
        return y;
    }

    let mut y = vec![0.0f32; out_dim];
    let chunk = out_dim.div_ceil(n_threads);
    std::thread::scope(|s| {
        // Each thread writes a disjoint slice of `y`. We split the
        // output vector into mutable subslices via `chunks_mut` so the
        // borrow checker can prove the writes don't alias.
        let mut handles = Vec::with_capacity(n_threads);
        for (t, y_chunk) in y.chunks_mut(chunk).enumerate() {
            let j0 = t * chunk;
            let w_ref = w;
            let x_ref = x;
            handles.push(s.spawn(move || {
                for (local_j, slot) in y_chunk.iter_mut().enumerate() {
                    let j = j0 + local_j;
                    let row = &w_ref[j * in_dim..(j + 1) * in_dim];
                    let mut acc = 0.0_f32;
                    for i in 0..in_dim {
                        acc += row[i] * x_ref[i];
                    }
                    *slot = acc;
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
    });
    y
}

/// Multi-head self-attention at S=1, with diagonal-only mask. At S=1
/// the softmax produces [1.0] exactly, so attn_out = v_proj(h)
/// effectively — we compute the full path here for parity-test clarity.
fn attention_s1(h: &[f32], blk: &TrainedBlock, n_heads: usize, hidden: usize) -> Vec<f32> {
    let head_dim = hidden / n_heads;
    // q = q_proj(h), k = k_proj(h), v = v_proj(h). All (hidden,).
    let q = matmul_no_bias(&blk.q_proj, h, hidden, hidden);
    let k = matmul_no_bias(&blk.k_proj, h, hidden, hidden);
    let v = matmul_no_bias(&blk.v_proj, h, hidden, hidden);
    // Per head, attn_out_head = softmax(q·k / sqrt(d) + mask=0) · v
    //                        = softmax([scalar]) · v = 1.0 * v = v
    // So the head-split + softmax + recombine is a no-op on v at S=1.
    // We still compute q·k and softmax explicitly for parity clarity.
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut attn_hidden = vec![0.0f32; hidden];
    for n in 0..n_heads {
        let off = n * head_dim;
        // q·k for this head (head_dim-vector dot product).
        let mut qk = 0.0_f32;
        for d in 0..head_dim {
            qk += q[off + d] * k[off + d];
        }
        // Scores: scalar = qk * scale. Mask = 0. softmax over 1
        // element is exactly 1.0; we don't need exp() for that.
        let _scores = qk * scale; // computed for parity clarity, but unused
        // attn_out for this head = 1.0 * v_head = v_head.
        for d in 0..head_dim {
            attn_hidden[off + d] = v[off + d];
        }
    }
    // out_proj projects (hidden,) → (hidden,).
    matmul_no_bias(&blk.out_proj, &attn_hidden, hidden, hidden)
}

/// SwiGLU: `down(silu(gate(h)) * up(h))`.
fn swiglu(
    h: &[f32],
    gate: &[f32],
    up: &[f32],
    down: &[f32],
    hidden: usize,
    ff: usize,
) -> Vec<f32> {
    let g = matmul_no_bias(gate, h, ff, hidden);
    let u = matmul_no_bias(up, h, ff, hidden);
    let mut activated = vec![0.0f32; ff];
    for i in 0..ff {
        // SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x)).
        let gi = g[i];
        let sigmoid = 1.0 / (1.0 + (-gi).exp());
        activated[i] = gi * sigmoid * u[i];
    }
    matmul_no_bias(down, &activated, hidden, ff)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rms_norm_unit_vector_is_unit() {
        // x = [1, 1, 1, 1] → ms = 1 → scale = 1 → result = weight elementwise.
        let x = vec![1.0_f32; 4];
        let w = vec![2.0_f32; 4];
        let y = rms_norm(&x, &w, RMS_EPS);
        for &yi in &y {
            assert!((yi - 2.0).abs() < 1e-5, "expected 2.0, got {yi}");
        }
    }

    #[test]
    fn matmul_identity_is_identity() {
        // W = I, x = [1,2,3] → y = [1,2,3].
        let w = vec![
            1.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, //
            0.0, 0.0, 1.0,
        ];
        let x = vec![1.0, 2.0, 3.0];
        let y = matmul_no_bias(&w, &x, 3, 3);
        assert_eq!(y, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn swiglu_zero_input_is_zero() {
        // h = 0 → gate(0) = 0 → silu(0) = 0 → 0 * up(0) = 0 → down(0) = 0.
        let h = vec![0.0_f32; 4];
        let gate = vec![1.0_f32; 8 * 4]; // ff=8
        let up = vec![1.0_f32; 8 * 4];
        let down = vec![1.0_f32; 4 * 8];
        let y = swiglu(&h, &gate, &up, &down, 4, 8);
        for &yi in &y {
            assert!(yi.abs() < 1e-6, "expected ~0, got {yi}");
        }
    }
}
